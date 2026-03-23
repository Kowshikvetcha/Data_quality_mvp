"""Centralized CSS and styled UI helpers for the application."""
import streamlit as st

# Color constants
CLEANING_COLOR = "#4A90D9"  # blue
ML_COLOR = "#7C3AED"  # purple


def inject_custom_css():
    """Inject custom CSS for visual polish. Call once at app startup."""
    st.markdown(
        """
        <style>
        /* Styled metric cards */
        div[data-testid="stMetric"] {
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 12px 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }

        /* Sidebar radio hover highlights */
        div[data-testid="stSidebar"] .stRadio label:hover {
            background-color: rgba(74, 144, 217, 0.08);
            border-radius: 4px;
        }

        /* Thinner dividers */
        hr {
            margin-top: 0.8rem;
            margin-bottom: 0.8rem;
            border-top: 1px solid #E2E8F0;
        }

        /* Rounded primary buttons */
        button[kind="primary"] {
            border-radius: 8px;
        }

        /* Section selectbox styling in sidebar */
        div[data-testid="stSidebar"] .stSelectbox > div > div {
            border-radius: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def styled_page_header(title, subtitle=None):
    """Render a page header with a colored left-border accent."""
    subtitle_html = (
        f'<p style="margin:0;color:#6B7280;font-size:0.95rem;">{subtitle}</p>'
        if subtitle
        else ""
    )
    st.markdown(
        f"""
        <div style="border-left:4px solid {CLEANING_COLOR};padding-left:12px;margin-bottom:1rem;">
            <h2 style="margin:0;padding:0;">{title}</h2>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def styled_section_header(title):
    """Render a lighter subheader with a bottom-border accent."""
    st.markdown(
        f"""
        <div style="border-bottom:2px solid {CLEANING_COLOR};padding-bottom:4px;margin-bottom:0.75rem;">
            <h4 style="margin:0;color:#374151;">{title}</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )
