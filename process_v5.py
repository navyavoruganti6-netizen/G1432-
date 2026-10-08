import json
from pathlib import Path

from inference import InferencePipeline


WORKSPACE = "venu-sudha"

WORKFLOW_ID = (
    "badminton-singles-players-"
    "vbadminton-singles-players-5-yolo26n-t1-logic"
)

INPUT_VIDEO = Path("video/badminton_test_10sec.mp4")

first_prediction_printed = False


def on_prediction(prediction, video_frame):

    global first_prediction_printed

    # --------------------------------------------------------
    # INSPECT FIRST COMPLETE PREDICTION
    # --------------------------------------------------------

    if not first_prediction_printed:

        print()
        print("=" * 80)
        print("FULL V5 WORKFLOW OUTPUT")
        print("=" * 80)

        print(
            "Prediction type:",
            type(prediction)
        )

        if isinstance(prediction, dict):

            print(
                "Top-level keys:",
                list(prediction.keys())
            )

            for key, value in prediction.items():

                print()
                print("-" * 80)
                print(f"FIELD: {key}")
                print("-" * 80)

                print(
                    "Type:",
                    type(value)
                )

                print(
                    "Value:"
                )

                try:
                    print(
                        json.dumps(
                            value,
                            indent=2,
                            default=str
                        )
                    )

                except Exception:
                    print(
                        str(value)
                    )

        else:

            print(
                prediction
            )

        print()
        print("=" * 80)
        print("END FULL V5 WORKFLOW OUTPUT")
        print("=" * 80)

        first_prediction_printed = True

    # --------------------------------------------------------
    # NORMAL PLAYER COUNT
    # --------------------------------------------------------

    player_count = 0

    if isinstance(prediction, dict):

        detections = prediction.get(
            "predictions"
        )

        if detections is not None:

            try:

                class_names = detections.data.get(
                    "class_name"
                )

                for class_name in class_names:

                    if str(class_name) == "player":
                        player_count += 1

            except Exception:
                pass

    frame_number = getattr(
        on_prediction,
        "frame_number",
        0
    )

    if frame_number % 30 == 0:

        print(
            f"Frame {frame_number}: "
            f"{player_count} player(s)"
        )

    on_prediction.frame_number = frame_number + 1


def main():

    print("=" * 80)
    print("BADMINTON AI - V5 WORKFLOW OUTPUT INSPECTION")
    print("=" * 80)

    print(
        f"Workspace : {WORKSPACE}"
    )

    print(
        f"Workflow  : {WORKFLOW_ID}"
    )

    print(
        f"Input     : {INPUT_VIDEO}"
    )

    print()

    if not INPUT_VIDEO.exists():

        print(
            "ERROR: Video not found:"
        )

        print(INPUT_VIDEO)

        return

    print(
        "Starting Roboflow workflow..."
    )

    print()

    pipeline = InferencePipeline.init_with_workflow(
        api_key=None,
        workspace_name=WORKSPACE,
        workflow_id=WORKFLOW_ID,
        video_reference=str(INPUT_VIDEO),
        max_fps=30,
        on_prediction=on_prediction,
    )

    try:

        pipeline.start()
        pipeline.join()

    except KeyboardInterrupt:

        print()
        print(
            "Stopped by user."
        )

    except Exception as e:

        print()
        print("=" * 80)
        print("PIPELINE ERROR")
        print("=" * 80)
        print(e)
        print("=" * 80)

        raise


if __name__ == "__main__":
    main()