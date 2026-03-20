import logging
from pathlib import Path
from time import perf_counter

from flask import Flask, jsonify, render_template, request

from inference.predict import predict_multimodal
from risk_engine.risk import score_paths
from utils.config import DEFAULT_MODEL_PATH


logger = logging.getLogger(__name__)


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).resolve().parent.parent / "templates"),
        static_folder=str(Path(__file__).resolve().parent.parent / "static"),
    )

    if DEFAULT_MODEL_PATH.exists():
        app.logger.info("Model checkpoint found at %s", DEFAULT_MODEL_PATH)
    else:
        app.logger.error("Model checkpoint not found at %s", DEFAULT_MODEL_PATH)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        model_ready = DEFAULT_MODEL_PATH.exists()
        return jsonify(
            {
                "status": "ok" if model_ready else "degraded",
                "model_ready": model_ready,
                "model_path": str(DEFAULT_MODEL_PATH.relative_to(Path(__file__).resolve().parent.parent)),
            }
        )

    @app.post("/predict")
    def predict():
        started_at = perf_counter()

        if not DEFAULT_MODEL_PATH.exists():
            app.logger.error("Prediction requested but model file is missing: %s", DEFAULT_MODEL_PATH)
            return (
                jsonify(
                    {
                        "error": "Model checkpoint not found.",
                        "model_path": str(DEFAULT_MODEL_PATH.relative_to(Path(__file__).resolve().parent.parent)),
                    }
                ),
                503,
            )

        try:
            payload = request.get_json(force=True) or {}
            trajectory = payload.get("trajectory", [])
            result = predict_multimodal(trajectory, model_path=DEFAULT_MODEL_PATH)
            result["risk"] = score_paths(result["paths"], result["probabilities"])
            result["meta"]["latency_ms"] = round((perf_counter() - started_at) * 1000, 2)
            return jsonify(result)
        except ValueError as exc:
            app.logger.warning("Invalid prediction payload: %s", exc)
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:  # pragma: no cover - deployment safety net
            app.logger.exception("Prediction failed: %s", exc)
            return jsonify({"error": "Prediction failed. Check server logs for details."}), 500

    return app


app = create_app()
