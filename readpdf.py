from flask import Blueprint, request, jsonify
import pdfplumber
from summarize import summarize_article
import notes_by_rank
       
api_blueprint = Blueprint("api_blueprint", __name__)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"pdf"}


@api_blueprint.route("/extract-text", methods=["POST"])
def extract_text():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected for uploading"}), 400

    if file and allowed_file(file.filename):
        try:
            with pdfplumber.open(file) as pdf:
                text = notes_by_rank.summarize_article(pdf)
            return jsonify({"text": text})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Allowed file types are pdf"}), 400


@api_blueprint.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Not found" + error}), 404


@api_blueprint.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error" + error}), 500
