"""Faster nnU-Net v1 inference for the Atlas models, on CUDA, Apple MPS or CPU.

Same preprocessing, sliding window (step 0.5, Gaussian weighting), fold averaging and export as
`nnUNet_predict`; only the device loop is replaced:
- patches are predicted and Gaussian-aggregated on the accelerator (nnU-Net v1 moves every patch to the CPU),
- the fold ensemble is accumulated on the accelerator, falling back to CPU RAM when it does not fit,
- the export resamples one class at a time on the accelerator with a running argmax, instead of resampling
  all classes on the CPU with skimage,
- the model is loaded once for the whole input folder.
On CPU in fp32 the output is bit-identical to `nnUNet_predict --disable_mixed_precision`.

usage (same folder convention as nnUNet_predict, CASE_0000.nii.gz in INPUT_FOLDER):
    python inference/predict_atlas.py -i INPUT_FOLDER -o OUTPUT_FOLDER -t 901          # Atlas dataset model (V1)
    python inference/predict_atlas.py -i INPUT_FOLDER -o OUTPUT_FOLDER -t 902          # Atlas prediction model (V2)
Options: -f 0 1 2 3 4 (default: all folds), --device cuda|mps|cpu (default: first available),
--disable_mixed_precision, --tta (mirroring; nnUNet_predict's default for V1, see README), -m MODEL_DIR.
"""
import argparse
import itertools
import os
import time

import numpy as np
import SimpleITK as sitk
import torch
from batchgenerators.augmentations.utils import pad_nd_image
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.preprocessing.preprocessing import get_do_separate_z, get_lowres_axis, resample_data_or_seg
from nnunet.training.model_restore import load_model_and_checkpoint_files

MODELS = {  # task -> (folder under $RESULTS_FOLDER/nnUNet/3d_fullres, checkpoint)
    901: ("Task901_Atlas_Dataset/nnUNetTrainerV2__nnUNetPlansv2.1", "model_final_checkpoint"),
    902: ("Task902_Anatomy_Prediction/nnUNetTrainerV2_noMirroring__nnUNetPlansv2.1", "model_latest"),
}


class Model:
    def __init__(self, trainer, params, folds):
        self.trainer, self.params, self.folds = trainer, params, folds


def load_model(model_dir, folds=None, checkpoint="model_final_checkpoint"):
    if folds is None:
        folds = sorted(int(d.split("_")[1]) for d in os.listdir(model_dir) if d.startswith("fold_"))
    trainer, params = load_model_and_checkpoint_files(model_dir, list(folds), mixed_precision=False,
                                                      checkpoint_name=checkpoint)
    return Model(trainer, params, list(folds))


def preprocess(model, inp):
    d, _, props = model.trainer.preprocess_patient([inp])
    return d, props


def predict(model, data, out, device="cuda", amp=True, tta=False, step=0.5, fp16=False, cpu_export=False):
    """Returns a dict of timings. amp = fp16 autocast for the network (nnU-Net v1 CUDA default)."""
    d, props = data
    trainer, params = model.trainer, model.params
    dev = torch.device(device)
    dtype = torch.float16 if fp16 else torch.float32
    t0 = time.time()
    patch = np.array(trainer.patch_size)
    net = trainer.network
    net.do_ds = False
    net.eval().to(dev, dtype)

    data, slicer = pad_nd_image(d, patch, "constant", {"constant_values": 0}, True, None)
    shape = data.shape[1:]
    steps = SegmentationNetwork._compute_steps_for_sliding_window(patch, shape, step)
    gauss = torch.from_numpy(SegmentationNetwork._get_gaussian(patch, 1. / 8)).to(dev, torch.float32)
    gauss[gauss == 0] = gauss[gauss != 0].min()
    x_all = torch.from_numpy(data).to(dev, dtype)[None]
    nc = trainer.num_classes
    # The running fold sum lives on the device when the device also holds the per-fold accumulator with room
    # to spare; otherwise on the CPU (same arithmetic, per-class transfers). nc * voxels * 4 bytes each.
    need = nc * int(np.prod(shape)) * 4
    tot_dev = dev
    if dev.type == "cuda":
        free, _ = torch.cuda.mem_get_info(dev)
        if 2 * need + (2 << 30) > free:
            tot_dev = torch.device("cpu")
    total = torch.zeros((nc,) + shape, device=tot_dev, dtype=torch.float32)
    mirror_axes = trainer.data_aug_params.get("mirror_axes", (0, 1, 2)) if tta else ()
    flips = [()] + [c for r in range(1, len(mirror_axes) + 1) for c in itertools.combinations(mirror_axes, r)]

    with torch.no_grad():
        for p in params:
            net.load_state_dict(p["state_dict"])
            net.to(dev, dtype)
            # fp32 accumulation: fp16 underflows at the Gaussian border (0/0 -> NaN)
            agg = torch.zeros((nc,) + shape, device=dev, dtype=torch.float32)
            cnt = torch.zeros(shape, device=dev, dtype=torch.float32)
            for x in steps[0]:
                for y in steps[1]:
                    for z in steps[2]:
                        sl = (slice(x, x + patch[0]), slice(y, y + patch[1]), slice(z, z + patch[2]))
                        xp = x_all[(slice(None), slice(None)) + sl]
                        pred = torch.zeros((nc,) + tuple(patch), device=dev, dtype=torch.float32)
                        for f in flips:
                            dims = [ax + 2 for ax in f]
                            inp = torch.flip(xp, dims) if f else xp
                            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                                logits = net(inp)
                            o = torch.softmax(logits.float(), 1)
                            pred += (torch.flip(o, dims) if f else o)[0]
                        pred /= len(flips)
                        agg[(slice(None),) + sl] += pred * gauss
                        cnt[sl] += gauss
            for c in range(nc):  # in place, one class at a time, to bound memory
                total[c] += (agg[c] / cnt).to(tot_dev)
            del agg, cnt
        total /= len(params)
    t1 = time.time()

    # Export identical to nnunet.inference.segmentation_export.save_segmentation_nifti_from_softmax
    # (order 1, force_separate_z None, order_z 0), but resampled one class at a time with a
    # running argmax (first index wins ties, like np.argmax) instead of holding all classes.
    crop = tuple(slicer[1:])
    target = props["size_after_cropping"]
    if get_do_separate_z(props["original_spacing"]):
        sep, axis = True, get_lowres_axis(props["original_spacing"])
    elif get_do_separate_z(props["spacing_after_resampling"]):
        sep, axis = True, get_lowres_axis(props["spacing_after_resampling"])
    else:
        sep, axis = False, None
    if axis is not None and len(axis) != 1:
        sep = False
    needs_resample = any(i != j for i, j in zip(total[0][crop].shape, target))
    if not sep and device != "cpu":
        # Trilinear (align_corners=False, edge clamp) on the device equals skimage resize
        # order 1 / mode edge / no anti-aliasing used by nnU-Net to float32 rounding (~1e-5).
        best_val = best_idx = None
        for c in range(nc):
            ch = total[c][crop].to(dev)[None, None]
            if needs_resample:
                ch = torch.nn.functional.interpolate(ch, size=tuple(int(t) for t in target), mode="trilinear",
                                                     align_corners=False)
            ch = ch[0, 0]
            if best_val is None:
                best_val, best_idx = ch.clone(), torch.zeros(ch.shape, dtype=torch.uint8, device=dev)
            else:
                m = ch > best_val
                best_val[m] = ch[m]
                best_idx[m] = c
        seg = best_idx.cpu().numpy()
    elif sep and device != "cpu" and not cpu_export:
        # Separate-z on the device: per-slice bilinear in-plane (== skimage order 1, mode edge) and nearest
        # along the low-resolution axis with the exact index map of nnU-Net's map_coordinates(order=0).
        from scipy.ndimage import map_coordinates
        ax = axis[0]
        others = [k for k in range(3) if k != ax]
        src = total[0][crop].shape
        new_plane = tuple(int(target[k]) for k in others)
        n_lo, t_lo = src[ax], int(target[ax])
        zc = (n_lo / t_lo) * (np.arange(t_lo) + 0.5) - 0.5
        zidx = map_coordinates(np.arange(n_lo, dtype=float), [zc], order=0, mode="nearest").round().astype(int)
        zidx_t = torch.from_numpy(zidx).to(dev)
        perm = [ax] + others
        inv = list(np.argsort(perm))
        best_val = best_idx = None
        for c in range(nc):
            ch = total[c][crop].to(dev).permute(*perm)
            if tuple(ch.shape[1:]) != new_plane:
                ch = torch.nn.functional.interpolate(ch[:, None], size=new_plane, mode="bilinear",
                                                     align_corners=False)[:, 0]
            if n_lo != t_lo:
                ch = ch.index_select(0, zidx_t)
            ch = ch.permute(*inv)
            if best_val is None:
                best_val, best_idx = ch.clone(), torch.zeros(ch.shape, dtype=torch.uint8, device=dev)
            else:
                m = ch > best_val
                best_val[m] = ch[m]
                best_idx[m] = c
        seg = best_idx.cpu().numpy()
    else:
        best_val = best_idx = None
        for c in range(nc):
            ch = total[c][crop].cpu().numpy()[None]
            if needs_resample:
                ch = resample_data_or_seg(ch, target, is_seg=False, axis=axis, order=1, do_separate_z=sep, order_z=0)
            ch = ch[0]
            if best_val is None:
                best_val, best_idx = ch.copy(), np.zeros(ch.shape, np.uint8)
            else:
                m = ch > best_val
                best_val[m] = ch[m]
                best_idx[m] = c
        seg = best_idx
    del total
    bbox = props.get("crop_bbox")
    if bbox is not None:
        full = np.zeros(props["original_size_of_raw_data"], dtype=np.uint8)
        for k in range(3):
            bbox[k][1] = min(bbox[k][0] + seg.shape[k], props["original_size_of_raw_data"][k])
        full[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]] = seg
        seg = full
    img = sitk.GetImageFromArray(seg.astype(np.uint8))
    img.SetSpacing(props["itk_spacing"])
    img.SetOrigin(props["itk_origin"])
    img.SetDirection(props["itk_direction"])
    sitk.WriteImage(img, out)
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return {"predict_s": round(t1 - t0, 1), "export_s": round(time.time() - t1, 1), "separate_z": bool(sep),
            "steps": [len(s) for s in steps], "folds": model.folds, "tta": bool(flips[1:]), "device": str(dev),
            "sum_on": str(tot_dev)}


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input_folder", required=True)
    ap.add_argument("-o", "--output_folder", required=True)
    ap.add_argument("-t", "--task", type=int, choices=sorted(MODELS), default=901)
    ap.add_argument("-m", "--model_dir", help="model folder with fold_0..fold_4 and plans.pkl "
                                              "(default: from $RESULTS_FOLDER and --task)")
    ap.add_argument("-chk", "--checkpoint", help="default: model_final_checkpoint (901), model_latest (902)")
    ap.add_argument("-f", "--folds", nargs="+", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--disable_mixed_precision", action="store_true")
    ap.add_argument("--tta", action="store_true", help="mirroring test-time augmentation (8x slower)")
    ap.add_argument("--step_size", type=float, default=0.5)
    a = ap.parse_args()
    sub, chk = MODELS[a.task]
    model_dir = a.model_dir or os.path.join(os.environ["RESULTS_FOLDER"], "nnUNet", "3d_fullres", sub)
    device = a.device or default_device()
    amp = device != "cpu" and not a.disable_mixed_precision
    os.makedirs(a.output_folder, exist_ok=True)
    cases = sorted(f[:-len("_0000.nii.gz")] for f in os.listdir(a.input_folder) if f.endswith("_0000.nii.gz"))
    model = load_model(model_dir, a.folds, a.checkpoint or chk)
    print(f"{len(cases)} cases, folds {model.folds}, device {device}, amp {amp}, tta {a.tta}", flush=True)
    for case in cases:
        out = os.path.join(a.output_folder, case + ".nii.gz")
        t = time.time()
        data = preprocess(model, os.path.join(a.input_folder, case + "_0000.nii.gz"))
        info = predict(model, data, out, device, amp=amp, tta=a.tta, step=a.step_size)
        print(case, {"total_s": round(time.time() - t, 1), **info}, flush=True)


if __name__ == "__main__":
    main()
