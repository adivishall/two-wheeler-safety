import os
import argparse

import requests

from modules.detector import load_models, analyze_image

# ---------------------------------
# SETTINGS
# ---------------------------------

DEFAULT_IMAGE_PATH = "plate8.jpeg"
DEFAULT_MODEL_PATH = "runs/detect/traffic_model-2/weights/best.pt"
DEFAULT_API_URL = "http://127.0.0.1:5000/detect"

# Must match DETECT_API_KEY on the server if it has one set; empty is
# fine when the server has no key configured.
API_KEY = os.environ.get("DETECT_API_KEY", "")

# ---------------------------------
# CLI ARGS
# ---------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect two-wheeler violations in a single image and report them to the fine checker API."
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE_PATH,
                         help=f"image file to scan (default: {DEFAULT_IMAGE_PATH})")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH,
                         help="path to the YOLO weights file")
    parser.add_argument("--api-url", default=DEFAULT_API_URL,
                         help="Flask /detect endpoint to report violations to")
    return parser.parse_args()


def main():
    args = parse_args()

    model, reader = load_models(args.model)

    result = analyze_image(args.image, model, reader)

    if result.get("error"):
        print(result["error"])
        return

    for det in result["detections"]:
        print("Detected:", det["label"])

    plate_number = result["plate"]
    if plate_number:
        print("Clean Plate:", plate_number)

    if result["evidence_file"]:
        print("Evidence Saved:", result["evidence_file"])

    # ---------------------------------
    # SEND TO WEBSITE
    # ---------------------------------

    if plate_number and result["violations"]:

        for violation in result["violations"]:

            try:
                response = requests.post(
                    args.api_url,
                    json={
                        "plate": plate_number,
                        "violation": violation,
                        "image_path": result["evidence_file"]
                    },
                    headers={"X-API-Key": API_KEY}
                )

                print("Sent:", violation)
                print(response.text)

            except requests.exceptions.ConnectionError:
                print(
                    f"Could not reach {args.api_url} — is app.py running?"
                )

    else:
        print("No valid violation found.")


if __name__ == "__main__":
    main()
