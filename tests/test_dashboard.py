"""
test_dashboard.py — Streamlit Dashboard integration tests.
Uses Streamlit's AppTest framework to verify app rendering, model selection,
and calibration states programmatically without requiring playwright.
"""

import os
import sys
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

@pytest.fixture
def app_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        "src", "deepvol", "app", "app_v2.py"
    )

def test_dashboard_startup_and_rendering(app_path):
    """Verify that the dashboard starts up successfully and renders main elements."""
    at = AppTest.from_file(app_path)
    at.run(timeout=10)
    
    # Check for exceptions
    assert not at.exception, f"App crashed during startup: {at.exception}"
    
    # Check title and subheaders
    assert len(at.title) > 0
    assert "Deep Volatility Model Zoo" in at.title[0].value
    
    # Verify model selector exists in sidebar
    model_sel = at.sidebar.selectbox("model_selector")
    assert model_sel is not None
    assert "Classic Heston" in model_sel.options

def test_dashboard_model_switching(app_path):
    """Verify switching between different models updates the sidebar state."""
    at = AppTest.from_file(app_path)
    at.run(timeout=10)
    
    # Select Classic Heston
    at.sidebar.selectbox("model_selector").select("Classic Heston").run()
    assert not at.exception
    
    # Select SABR
    at.sidebar.selectbox("model_selector").select("SABR").run()
    assert not at.exception
    
    # Select Neural SDE
    at.sidebar.selectbox("model_selector").select("Neural SDE").run()
    assert not at.exception
    
    # Select Schwartz-Smith
    at.sidebar.selectbox("model_selector").select("Schwartz-Smith (2-Factor)").run()
    assert not at.exception

def test_dashboard_synthetic_surface_generation(app_path):
    """Verify generating a target surface from sidebar presets works without errors."""
    at = AppTest.from_file(app_path)
    at.run(timeout=10)
    
    # Select SABR
    at.sidebar.selectbox("model_selector").select("SABR").run()
    assert not at.exception
    
    # Find the generation button and click it
    # We can locate buttons using at.button
    gen_buttons = [b for b in at.button if "Generate Target Surface" in b.label]
    assert len(gen_buttons) > 0
    
    # Click the button
    gen_buttons[0].click().run()
    assert not at.exception
    
    # Check that the target surface is successfully created
    assert "target_iv" in at.session_state
    assert at.session_state["active_model"] == "SABR"
