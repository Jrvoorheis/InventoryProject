"""Streamlit web dashboard entry point for Supply Chain Analytics Inventory Decision Agent."""

import importlib
import app
importlib.reload(app)

app.run_streamlit_app()
