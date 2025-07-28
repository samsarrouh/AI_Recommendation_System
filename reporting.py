# reporting.py
import io
import base64
import logging
import pdfkit
import matplotlib.pyplot as plt
import pandas as pd

from db import global_collection as collection
from ai_utils import get_embedding, ollama_generate_response

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def aggregate_enriched_data(recommendation):
    embedding = get_embedding(recommendation)
    if not embedding:
        return "", {}
    try:
        results = collection.query(query_embeddings=[embedding], n_results=10)
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        aggregated_text = " ".join(docs)
        aggregated_metadata = {}
        for meta in metadatas:
            for key, value in meta.items():
                aggregated_metadata.setdefault(key, []).append(value)
        aggregated_metadata_counts = {}
        for key, values in aggregated_metadata.items():
            # Example: count each unique value
            counts = pd.Series(values).value_counts().to_dict()
            aggregated_metadata_counts[key] = counts
        return aggregated_text, aggregated_metadata_counts
    except Exception as e:
        logging.error(f"Error aggregating enriched data: {e}")
        return "", {}

def generate_visualization_from_metadata(aggregated_metadata):
    if not aggregated_metadata:
        return ""
    try:
        excluded_keys = {"filename", "source", "product_name"}
        candidate_key = None
        for key in aggregated_metadata.keys():
            if key not in excluded_keys:
                candidate_key = key
                break
        if not candidate_key:
            return ""
        counts = aggregated_metadata[candidate_key]
        categories = list(counts.keys())
        values = list(counts.values())
        plt.figure(figsize=(6, 4))
        plt.bar(categories, values, color='skyblue')
        plt.xlabel(candidate_key)
        plt.ylabel("Count")
        plt.title(f"Distribution of {candidate_key}")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        logging.error(f"Error generating visualization from metadata: {e}")
        return ""

def generate_visualization_from_dataset(dataset_text, chart_type):
    try:
        from io import StringIO
        df = pd.read_csv(StringIO(dataset_text))
        plt.figure(figsize=(6, 4))
        if chart_type == "bar":
            x = df.iloc[:, 0]
            y = df.iloc[:, 1]
            plt.bar(x, y, color='skyblue')
            plt.xlabel(x.name)
            plt.ylabel(y.name)
            plt.title("Bar Chart")
        elif chart_type == "line":
            x = df.iloc[:, 0]
            y = df.iloc[:, 1]
            plt.plot(x, y, marker='o')
            plt.xlabel(x.name)
            plt.ylabel(y.name)
            plt.title("Line Chart")
        elif chart_type == "scatter":
            x = df.iloc[:, 0]
            y = df.iloc[:, 1]
            plt.scatter(x, y, color='red')
            plt.xlabel(x.name)
            plt.ylabel(y.name)
            plt.title("Scatter Plot")
        elif chart_type == "wordcloud":
            from wordcloud import WordCloud
            text = " ".join(df.iloc[:, 0].astype(str).tolist())
            wc = WordCloud(width=600, height=400, background_color='white').generate(text)
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title("Word Cloud")
        else:
            plt.close()
            return ""
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        logging.error(f"Error generating visualization from dataset: {e}")
        return ""

def generate_block_analysis(block, recommendation, summary):
    block_html = ""
    # Generate a chart only if a CSV file was uploaded.
    csv_content = block.get("csv_content", "").strip()
    if csv_content:
        chart_type = block.get("chart_type", "bar").strip()
        visualization_img = generate_visualization_from_dataset(csv_content, chart_type)
        prompt = f"Based on the following user-supplied dataset (CSV format):\n{csv_content[:500]}...\n"
        prompt += "Provide a detailed analysis focusing on trends, key metrics, and actionable recommendations for data center power management."
        user_analysis = ollama_generate_response(prompt, context="")
        if not user_analysis:
            user_analysis = "No additional analysis available for user dataset."
        block_html += f"""
        <div class="movable">
          <h3>User-Supplied Visualization ({chart_type.capitalize()} Chart)</h3>
          <img src="data:image/png;base64,{visualization_img}" alt="Visualization">
        </div>
        <div class="movable">
          <h3>User-Supplied Data Analysis</h3>
          <p>{user_analysis}</p>
        </div>
        """
    else:
        # If no CSV is uploaded, do not generate a chart.
        prompt = f"Based on the following information:\nRecommendation: {recommendation}\nSummary: {summary}\n"
        prompt += "Generate a detailed report analysis that includes key insights, trends, and actionable recommendations."
        logging.info(f"Auto mode AI prompt: {prompt}")
        auto_analysis = ollama_generate_response(prompt, context="")
        if not auto_analysis:
            auto_analysis = "No additional analysis available."
        block_html += f"""
        <div class="movable">
          <h3>Automatically Generated Analysis</h3>
          <p>{auto_analysis}</p>
        </div>
        """
    # Include uploaded image if available.
    if block.get("image_content", "").strip():
        block_html += f"""
        <div class="movable">
          <h3>Uploaded Image</h3>
          <img src="{block.get('image_content')}" alt="Uploaded Image">
        </div>
        """
    return block_html



def generate_report_preview(report_config):
    recommendation = report_config.get("recommendation", "").strip()
    summary = report_config.get("summary", "").strip()
    if not recommendation or not summary:
        raise ValueError("Recommendation and summary must be provided for report generation.")

    blocks = report_config.get("blocks", [])
    if not blocks:
        blocks = [{"block_type": "auto"}]

    blocks_html = ""
    for block in blocks:
        blocks_html += generate_block_analysis(block, recommendation, summary)

    html_report = f"""
    <html>
      <head>
        <style>
          body {{ font-family: Arial, sans-serif; padding: 20px; }}
          h2 {{ color: #2c3e50; }}
          .section {{ margin-bottom: 20px; }}
        </style>
      </head>
      <body>
        <h2>Detailed Report Analysis</h2>
        <div class="section">
          <h3>Selected Recommendation</h3>
          <p>{recommendation}</p>
        </div>
        <div class="section">
          <h3>Summary</h3>
          <p>{summary}</p>
        </div>
        {blocks_html}
      </body>
    </html>
    """
    return html_report, {}, None

def generate_report_pdf(report_config):
    html_report, _, _ = generate_report_preview(report_config)
    try:
        wkhtmltopdf_path = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
        config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
        pdf_data = pdfkit.from_string(html_report, False, configuration=config)
        return pdf_data
    except Exception as e:
        logging.error(f"Error generating PDF: {e}")
        return None

def generate_report_csv(report_config):
    # For CSV, we could output aggregated metadata or block data.
    # For now, just a placeholder.
    return "CSV export not implemented for multiple blocks yet."
