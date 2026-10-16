"""Live, opt-in BoundingBox proof; never run by the unit suite.

Run from the repository: python scripts/validate_bounding_boxes.py --output /tmp/boxes
Requires Pillow, TYPESAFE_API_KEY, and optionally cmpnd + CMPND_API_KEY for tracing.
Downloads public COCO validation images and annotations; sends images to TypeSafe.
Use --azure-model openai/DEPLOYMENT to exercise an Azure OpenAI-compatible v1
endpoint with AZURE_API_BASE and AZURE_API_KEY. All quality calls bypass DSPy's
cache.

This is a fixed, deliberately varied smoke sample, NOT COCO mAP or a random
estimate of provider quality. Selection precedes inference and failures remain
in results.json. No ground-truth coordinates are supplied to the model.
"""

import argparse
import asyncio
import base64
import hashlib
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import requests
from PIL import Image, ImageDraw, ImageOps

import dspy
from dspy._vendor.lm15.serde import response_to_dict

ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/eembc/mlmark/656d136cdbc1f11c468131be957ff5aa57b09734/"
    "datasets/COCO2017/annotations/instances_val2017.json"
)
ANNOTATIONS_SHA256 = "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
# First 12 distinct classes by ascending image id, one annotation for the class,
# non-crowd, dimensions >=280, box area between 3% and 70%; largest eligible mask.
# One annotation does not imply only one visible instance (notably the apples).
ANNOTATION_IDS = [2204286, 315724, 442619, 332341, 1177525, 324614,
                  170940, 308085, 46386, 165069, 1047376, 1085101]


def fetch(url, path):
    if not path.exists():
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path.read_bytes()


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def xyxy(box, width, height):
    return [box.left * width, box.top * height, box.right * width, box.bottom * height]


def cases(directory):
    raw = fetch(ANNOTATIONS_URL, directory / "instances_val2017.json")
    assert hashlib.sha256(raw).hexdigest() == ANNOTATIONS_SHA256
    data = json.loads(raw)
    images = {im["id"]: im for im in data["images"]}
    annotations = {a["id"]: a for a in data["annotations"]}
    categories = {c["id"]: c["name"] for c in data["categories"]}
    result = []
    for index, annotation_id in enumerate(ANNOTATION_IDS):
        annotation = annotations[annotation_id]
        info = images[annotation["image_id"]]
        url = f"https://s3.amazonaws.com/images.cocodataset.org/val2017/{info['file_name']}"
        path = directory / info["file_name"]
        fetch(url, path)
        image = Image.open(path).convert("RGB")
        assert image.size == (info["width"], info["height"])
        x, y, w, h = annotation["bbox"]
        target = categories[annotation["category_id"]]
        item = {
            "id": f"coco-{info['id']}", "image": image, "path": path, "target": target,
            "targets": {"box": (f"The {target} in inputs.photo.", [x, y, x + w, y + h])},
            "source": {"url": url, "annotation_id": annotation_id, "image_id": info["id"],
                       "license": info["license"], "bbox_xywh": annotation["bbox"]},
        }
        result.append(item)
        if index in (0, 3, 5, 8):
            flipped = ImageOps.mirror(image)
            flip_path = directory / f"flipped-{info['file_name']}"
            flipped.save(flip_path)
            result.append({**item, "id": item["id"] + "-flip", "image": flipped, "path": flip_path,
                           "targets": {"box": (item["targets"]["box"][0], [image.width-x-w, y, image.width-x, y+h])},
                           "source": {**item["source"], "transform": "horizontal flip"}})
    # Multiple questions against the same natural image, before any predictions.
    first = result[0]
    tv = next(a for a in data["annotations"] if a["image_id"] == 139 and categories[a["category_id"]] == "tv")
    x, y, w, h = tv["bbox"]
    result.append({**first, "id": "coco-139-multiple", "targets": {
        **first["targets"], "television": ("The television in inputs.photo.", [x, y, x+w, y+h]),
    }, "source": {**first["source"], "additional_annotation_id": tv["id"]}})
    # Exact asymmetric controls make axis swaps, image confusion, and boundary
    # errors observable independently of ambiguous natural-image annotations.
    for width, height, name in [(640, 400, "landscape"), (400, 640, "portrait")]:
        image = Image.new("RGB", (width, height), "white")
        red = [round(width*.55), round(height*.12), width, round(height*.57)]
        blue = [round(width*.06), round(height*.64), round(width*.29), round(height*.92)]
        if name == "portrait":
            red, blue = blue, red
        draw = ImageDraw.Draw(image)
        draw.rectangle((red[0], red[1], red[2]-1, red[3]-1), fill="red")
        draw.ellipse((blue[0], blue[1], blue[2]-1, blue[3]-1), fill="blue")
        path = directory / f"{name}.png"
        image.save(path)
        result.append({"id": f"synthetic-{name}", "image": image, "path": path, "target": "red rectangle",
                       "targets": {"box": ("The red rectangle in inputs.photo.", red),
                                   "blue": ("The blue ellipse in inputs.photo.", blue)},
                       "source": {"generated": True, "dimensions": [width, height]}})
    return result


def signature_for(case, *, mixed=True):
    fields = {"photo": (dspy.Image, dspy.InputField())}
    for name, (description, _) in case["targets"].items():
        fields[name] = (dspy.BoundingBox, dspy.OutputField(desc=description))
    if mixed:
        fields.update({
            "visible": (bool, dspy.OutputField(desc=f"Is there a {case['target']} in inputs.photo?")),
            "unicorn": (bool, dspy.OutputField(desc="Is there a real unicorn in inputs.photo?")),
            "category": (Literal[case["target"], "unicorn", "none"], dspy.OutputField(desc="Which of these is visible in inputs.photo?")),
            "size": (Literal[0, 1, 2], dspy.OutputField(desc="Size of the target: 0 small, 1 medium, 2 large.")),
        })
    return dspy.Signature(fields, "Locate the named targets in the image. Return tight boxes in normalized coordinates.")


async def run_case(lm, adapter, case, output, *, asynchronous=False, mixed=True, extra_inputs=None):
    signature = signature_for(case, mixed=mixed)
    inputs = {"photo": dspy.Image.from_path(str(case["path"])), **(extra_inputs or {})}
    for name in extra_inputs or {}:
        signature = signature.insert(0, name, dspy.InputField(), type_=dspy.Image)
    record = {"id": case["id"], "source": case["source"], "model": lm.model,
              "async": asynchronous, "mixed": mixed,
              "input_sha256": {name: hashlib.sha256(base64.b64decode(value.url.split(",", 1)[1])).hexdigest()
                               for name, value in inputs.items()},
              "questions": {name: field.json_schema_extra["desc"] for name, field in signature.output_fields.items()},
              "dimensions": case["image"].size, "started": datetime.now(timezone.utc).isoformat()}
    start = time.perf_counter()
    try:
        with dspy.context(adapter=adapter):
            predict = dspy.Predict(signature)
            result = await predict.acall(lm=lm, **inputs) if asynchronous else predict(lm=lm, **inputs)
        record["latency_s"] = time.perf_counter() - start
        response = lm.history[-1]["response"]
        record["response"] = response_to_dict(response, include_provider_data=True)
        record["prediction"] = result.toDict()
        record["metrics"] = {}
        image = case["image"].copy()
        draw = ImageDraw.Draw(image)
        for name, (_, truth) in case["targets"].items():
            box = result[name]
            assert isinstance(box, dspy.BoundingBox)
            if isinstance(adapter, dspy.TypeSafeAdapter):
                assert box.model_dump() == response.provider_data["typesafe"]["answers"][name]["box"]
                assert name not in (response.probabilities or {})
            pixels = xyxy(box, *image.size)
            record["metrics"][name] = {"truth_xyxy": truth, "predicted_xyxy": pixels, "iou": iou(pixels, truth),
                                      "max_edge_error": max(abs(a-b)/scale for a, b, scale in zip(pixels, truth, [image.width, image.height]*2, strict=True))}
            draw.rectangle(truth, outline="#00b857", width=3)
            draw.rectangle(pixels, outline="#e40075", width=2)
        if mixed and isinstance(adapter, dspy.TypeSafeAdapter):
            assert set(response.probabilities) == {"visible", "unicorn", "category", "size"}
            record["presence_correct"] = result.visible and not result.unicorn
            record["category_correct"] = result.category == case["target"]
        label = f"{case['id']} | " + " / ".join(f"{k} IoU {v['iou']:.3f}" for k, v in record["metrics"].items())
        canvas = Image.new("RGB", (image.width, image.height + 45), "#ffffff")
        canvas.paste(image, (0, 45))
        ImageDraw.Draw(canvas).text((8, 4), label + "\nGreen: ground truth. Pink: model.", fill="black")
        canvas.save(output / f"{case['id']}.png")
        record["contract_pass"] = True
    except Exception as exc:
        record["latency_s"] = time.perf_counter() - start
        record["contract_pass"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps({k: record[k] for k in ("id", "contract_pass", "latency_s")} | {"metrics": record.get("metrics"), "error": record.get("error")}), flush=True)
    return record


async def controls(samples, output):
    """Exercise documented follow-up workflows, not just box generation."""
    results = {}
    lm = dspy.LM("typesafe:jev-1.14.0", engine="lm15", cache=True, num_retries=0, timeout=60)
    primary = samples[-1]
    image = dspy.Image.from_path(str(primary["path"]))
    signature = signature_for(primary)
    predict = dspy.Predict(signature)
    try:
        with dspy.context(adapter=dspy.TypeSafeAdapter()):
            start = time.perf_counter()
            first = predict(photo=image, lm=lm)
            cold = time.perf_counter() - start
            response = lm.history[-1]["response"]
            cold_usage = lm.history[-1]["usage"]
            start = time.perf_counter()
            cached = predict(photo=image, lm=lm)
            warm = time.perf_counter() - start
            assert first.toDict() == cached.toDict()
            assert response.provider_data == lm.history[-1]["response"].provider_data
            assert cold_usage and lm.history[-1]["usage"] == {}
            # DSPy's existing sync and async cache namespaces are separate.
            async_first = await predict.acall(photo=image, lm=lm)
            assert lm.history[-1]["usage"]
            async_response = lm.history[-1]["response"]
            async_cached = await predict.acall(photo=image, lm=lm)
            assert async_first.toDict() == async_cached.toDict()
            assert async_response.provider_data == lm.history[-1]["response"].provider_data
            assert lm.history[-1]["usage"] == {}
            # New content at the same path must not reuse the previous answer.
            mutable = output / "inputs" / "mutable.png"
            primary["image"].save(mutable)
            old = dspy.Image.from_path(str(mutable))
            samples[-2]["image"].save(mutable)
            changed = dspy.Image.from_path(str(mutable))
            assert old.url != changed.url
            predict(photo=changed, lm=lm)
            assert lm.history[-1]["usage"]
            results["cache"] = {"cold_s": cold, "cached_s": warm, "same_answers_and_evidence": True,
                                "changed_image_misses": True, "separate_async_cache_verified": True}
            # Positive and negative edge controls: the portrait rectangle spans
            # x=[24,116], y=[410,589]. Crop removes top/left only, then resize.
            cropped = primary["image"].crop((40, 440, 160, 640)).resize((360, 600))
            cropped.save(output / "edge-crop.png")
            edge_signature = dspy.Signature({
                "photo": (dspy.Image, dspy.InputField()),
                **{edge: (bool, dspy.OutputField(desc=f"Does the red rectangle in inputs.photo continue past the {edge} edge of the image?"))
                   for edge in ("top", "bottom", "left", "right")},
            })
            edge_result = await dspy.Predict(edge_signature).acall(photo=dspy.Image(cropped), lm=lm)
            results["edge_questions"] = {"expected": {"top": True, "bottom": False, "left": True, "right": False},
                                         "actual": edge_result.toDict(),
                                         "response": response_to_dict(lm.history[-1]["response"], include_provider_data=True)}
            # Real absent-target behavior: Box has no null answer. Guard it with
            # Noul instead of interpreting any returned rectangle as detection.
            absent = dspy.Signature({"photo": (dspy.Image, dspy.InputField()),
                "visible": (bool, dspy.OutputField(desc="Is there a green triangle in inputs.photo?")),
                "box": (dspy.BoundingBox, dspy.OutputField(desc="The green triangle in inputs.photo."))})
            try:
                result = dspy.Predict(absent)(photo=image, lm=lm)
                results["absent"] = {"prediction": result.toDict(), "ignore_box": not result.visible}
            except dspy.AdapterParseError as exc:
                results["absent"] = {"invalid_box_rejected": True, "error": str(exc)}
            results["absent"]["response"] = response_to_dict(lm.history[-1]["response"], include_provider_data=True)
            # A target specified by a reference picture rather than its color.
            template = primary["image"].crop((10, 380, 140, 620)).resize((280, 517))
            match = {**primary, "id": "reference-match", "targets": {
                "box": ("The shape in inputs.photo that matches the shape in inputs.reference.", primary["targets"]["box"][1]),
            }}
            results["reference_match"] = await run_case(lm, dspy.TypeSafeAdapter(), match, output, mixed=False,
                                                       extra_inputs={"reference": dspy.Image(template)})
            # Caller-selected left-half tile of the difficult flipped phone.
            # Resize to satisfy Jev's 280px minimum side; scale all edges back
            # before comparing with the original annotation. No truth is sent.
            phone = next(case for case in samples if case["id"] == "coco-1296-flip")
            tile = phone["image"].crop((0, 0, 214, 640)).resize((428, 1280))
            tile_path = output / "inputs" / "phone-tile.png"
            tile.save(tile_path)
            tile_case = {**phone, "id": "phone-tile", "image": tile, "path": tile_path,
                         "targets": {"box": (phone["targets"]["box"][0], [2*v for v in phone["targets"]["box"][1]])},
                         "source": {**phone["source"], "crop_xyxy": [0, 0, 214, 640], "resize": [428, 1280]}}
            results["crop_refinement"] = await run_case(lm, dspy.TypeSafeAdapter(), tile_case, output, mixed=False)
            if results["crop_refinement"]["contract_pass"]:
                mapped = [v/2 for v in results["crop_refinement"]["metrics"]["box"]["predicted_xyxy"]]
                results["crop_refinement"]["original_image_xyxy"] = mapped
                results["crop_refinement"]["original_image_iou"] = iou(mapped, phone["targets"]["box"][1])
    finally:
        (output / "controls.json").write_text(json.dumps(results, indent=2))
        await lm.aclose()
    return results


async def main(args):
    dspy.configure_cache(enable_disk_cache=False)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    data_dir = output / "inputs"
    data_dir.mkdir(exist_ok=True)
    if os.getenv("CMPND_API_KEY"):
        import cmpnd
        cmpnd.configure(project_tags="dspy-bounding-box-proof")
        cmpnd.auto_instrument()
    samples = cases(data_dir)
    records = []
    lm = dspy.LM("typesafe:jev-1.14.0", engine="lm15", cache=False, num_retries=0, timeout=60)
    try:
        for index, case in enumerate(samples):
            records.append(await run_case(lm, dspy.TypeSafeAdapter(), case, output, asynchronous=index % 2 == 1))
            (output / "results.json").write_text(json.dumps(records, indent=2))
        case = {**samples[-1], "id": "box-only", "targets": {"box": samples[-1]["targets"]["box"]}}
        records.append(await run_case(lm, dspy.TypeSafeAdapter(), case, output, mixed=False))
        # Distinct landscape/portrait content tests image identity in one state.
        for primary, distractor in [(samples[-2], samples[-1]), (samples[-1], samples[-2])]:
            case = {**primary, "id": "two-images-" + primary["id"],
                    "targets": {"box": primary["targets"]["box"]}}
            records.append(await run_case(lm, dspy.TypeSafeAdapter(), case, output, asynchronous=True, mixed=False,
                                          extra_inputs={"reference": dspy.Image.from_path(str(distractor["path"]))}))
    finally:
        await lm.aclose()
    (output / "results.json").write_text(json.dumps(records, indent=2))
    control_results = await controls(samples, output)
    control_records = [control_results[name] for name in ("reference_match", "crop_refinement")]
    if args.azure_model:
        lm = dspy.LM(args.azure_model, engine="lm15", api_base=os.environ["AZURE_API_BASE"],
                     api_key=os.environ["AZURE_API_KEY"], cache=False, num_retries=0, max_tokens=16000, temperature=1)
        try:
            for index in (0, 3, -1):
                case = {**samples[index], "id": "azure-" + samples[index]["id"]}
                records.append(await run_case(lm, dspy.JSONAdapter(), case, output, asynchronous=index == 3))
        finally:
            await lm.aclose()
    (output / "results.json").write_text(json.dumps(records, indent=2))
    metrics = [m["iou"] for r in records if r["id"].startswith("coco-") for m in r.get("metrics", {}).values()]
    summary = {"calls": len(records), "contract_pass": sum(r["contract_pass"] for r in records),
               "control_calls": len(control_records), "control_contract_pass": sum(r["contract_pass"] for r in control_records),
               "coco_boxes": len(metrics), "coco_mean_iou": statistics.mean(metrics) if metrics else None,
               "coco_iou_ge_0.5": sum(v >= .5 for v in metrics), "coco_iou_ge_0.75": sum(v >= .75 for v in metrics)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    if os.getenv("CMPND_API_KEY"):
        cmpnd.flush_exporter()
    if not all(record["contract_pass"] for record in [*records, *control_records]):
        raise SystemExit("Integration failures recorded in results.json or controls.json; inspect them before trusting quality metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--azure-model")
    asyncio.run(main(parser.parse_args()))
