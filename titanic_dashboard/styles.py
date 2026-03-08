"""CSS styling for Titanic Dashboard"""

import streamlit as st


def apply_dark_theme():
    """Apply dark ocean theme to the dashboard"""

    st.markdown(
        """
        <style>

        /* ----------------------------------------------------
        MAIN BACKGROUND
        ---------------------------------------------------- */

        .stApp {
            background: linear-gradient(
                135deg,
                #0f2027 0%,
                #203a43 50%,
                #2c5364 100%
            );
            color: white;
        }

        /* ----------------------------------------------------
        HEADINGS
        ---------------------------------------------------- */

        h1, h2, h3 {
            color: #ffffff !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }

        /* ----------------------------------------------------
        METRIC CARDS
        ---------------------------------------------------- */

        .metric-card {
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
            box-shadow: 0 8px 20px rgba(0,0,0,0.25);
        }

        /* ----------------------------------------------------
        BUTTONS
        ---------------------------------------------------- */

        .stButton>button {
            background: linear-gradient(135deg,#00b4db,#0083b0);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 12px 28px;
            font-weight: 600;
            transition: all 0.25s ease;
        }

        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(0,0,0,0.3);
        }

        /* ----------------------------------------------------
        SIDEBAR
        ---------------------------------------------------- */

        section[data-testid="stSidebar"] {
            background: rgba(15,32,39,0.85);
            backdrop-filter: blur(8px);
        }

        /* ----------------------------------------------------
        INSIGHT BOXES
        ---------------------------------------------------- */

        .insight-box {
            background: rgba(46,204,113,0.15);
            border-left: 4px solid #2ecc71;
            padding: 15px;
            border-radius: 0 10px 10px 0;
            margin: 10px 0;
        }

        .warning-box {
            background: rgba(231,76,60,0.15);
            border-left: 4px solid #e74c3c;
            padding: 15px;
            border-radius: 0 10px 10px 0;
            margin: 10px 0;
        }

        .ml-box {
            background: rgba(155,89,182,0.15);
            border-left: 4px solid #9b59b6;
            padding: 15px;
            border-radius: 0 10px 10px 0;
            margin: 10px 0;
        }

        /* ----------------------------------------------------
        TABS
        ---------------------------------------------------- */

        .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.05);
            border-radius: 10px 10px 0 0;
            padding: 12px 24px;
            color: white;
            font-weight: 600;
        }

        .stTabs [aria-selected="true"] {
            background: rgba(0,180,219,0.25);
        }

        /* ----------------------------------------------------
        SCROLLBAR
        ---------------------------------------------------- */

        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-thumb {
            background: #00b4db;
            border-radius: 5px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(255,255,255,0.05);
        }
        [data-testid="stMetricLabel"] {
        color: #e6e6e6 !important;
        font-weight: 600;
        }

        [data-testid="stMetricValue"] {
        color: #ffffff !important;
        }
        [data-testid="stMetricDelta"] {
        color: #00e676 !important;
        }
        
        
        
        
        
        </style>
        """,
        unsafe_allow_html=True,
    )