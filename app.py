"""Streamlit app for FIFA World Cup ML Analytics."""

from __future__ import annotations

import logging
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from main import load_object, predict_custom_match


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
RAW_DATA_FILE = DATA_DIR / "fifa_world_cup_2026_player_performance-selected-columns-2.csv"
VIEW_QUERY_PARAM = "view"
VIEW_OPTIONS = [
    "Match Predictor",
    "Player Clusters",
    "Player Explorer",
    "Model Insights",
]
LOGGER = logging.getLogger(__name__)


@st.cache_resource
def load_model_artifacts():
    model = load_object(MODELS_DIR / "final_match_model.pkl")
    training_columns = load_object(MODELS_DIR / "training_columns.pkl")
    return model, training_columns


@st.cache_data
def load_player_data():
    profiles = pd.read_csv(DATA_DIR / "player_profiles.csv")
    clustered_path = DATA_DIR / "player_profiles_with_clusters.csv"
    clustered = pd.read_csv(clustered_path) if clustered_path.exists() else pd.DataFrame()

    roster = pd.read_csv(
        RAW_DATA_FILE,
        usecols=["team", "player_name", "position", "jersey_number"],
    )
    roster = (
        roster.dropna(subset=["team", "player_name"])
        .sort_values(["team", "player_name", "position", "jersey_number"])
        .groupby(["team", "player_name", "position"], as_index=False)["jersey_number"]
        .first()
    )

    merge_keys = ["team", "player_name", "position"]
    if "jersey_number" not in profiles.columns:
        profiles = profiles.merge(roster, on=merge_keys, how="left")
    if not clustered.empty and "jersey_number" not in clustered.columns:
        clustered = clustered.merge(roster, on=merge_keys, how="left")
    return profiles, clustered


@st.cache_data
def load_match_count() -> int:
    matches = pd.read_csv(RAW_DATA_FILE, usecols=["match_id"])
    return int(matches["match_id"].nunique())


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --bg: #05070D;
          --surface: #10141F;
          --surface-2: #171C2A;
          --surface-3: #22293A;
          --text: #FFFFFF;
          --muted: #A7B0C0;
          --muted-2: #697386;
          --gold: #D6A84F;
          --soft-gold: #F4D58D;
          --green: #22C55E;
          --amber: #F59E0B;
          --red: #EF4444;
          --blue: #3B82F6;
          --pink: #F43F5E;
        }

        html, body, [data-testid="stAppViewContainer"] {
          background:
            radial-gradient(circle at 10% 8%, rgba(214, 168, 79, 0.16), transparent 28rem),
            radial-gradient(circle at 86% 12%, rgba(59, 130, 246, 0.13), transparent 24rem),
            linear-gradient(135deg, #05070D 0%, #09101E 48%, #05070D 100%);
          color: var(--text);
          overflow-x: hidden;
        }

        [data-testid="stHeader"] {
          background: rgba(5, 7, 13, 0);
        }

        [data-testid="stMainBlockContainer"] {
          padding-top: 1.2rem;
          max-width: 1280px;
        }

        h1, h2, h3 {
          font-family: "Avenir Next Condensed", "DIN Condensed", "Arial Narrow", sans-serif;
          letter-spacing: 0;
        }

        p, label, span, div {
          font-family: "Avenir Next", "Trebuchet MS", sans-serif;
        }

        .hero {
          position: relative;
          overflow: hidden;
          border: 1px solid rgba(214, 168, 79, 0.28);
          border-radius: 8px;
          padding: 34px 36px;
          background:
            linear-gradient(115deg, rgba(16, 20, 31, 0.95), rgba(23, 28, 42, 0.88)),
            repeating-linear-gradient(90deg, rgba(255,255,255,0.035) 0, rgba(255,255,255,0.035) 1px, transparent 1px, transparent 58px);
          box-shadow: 0 24px 70px rgba(0, 0, 0, 0.42);
        }

        .hero:after {
          content: "26";
          position: absolute;
          right: 30px;
          top: -38px;
          font-family: "Avenir Next Condensed", "DIN Condensed", sans-serif;
          font-size: 188px;
          font-weight: 900;
          line-height: 1;
          color: rgba(214, 168, 79, 0.08);
        }

        .eyebrow {
          color: var(--soft-gold);
          font-size: 0.78rem;
          font-weight: 800;
          letter-spacing: 0.12em;
          text-transform: uppercase;
        }

        .hero-title {
          margin: 0.2rem 0 0.7rem;
          max-width: 760px;
          color: var(--text);
          font-size: clamp(2.6rem, 6vw, 5.5rem);
          line-height: 0.92;
          text-transform: uppercase;
        }

        .hero-copy {
          max-width: 780px;
          color: var(--muted);
          font-size: 1.05rem;
          line-height: 1.6;
        }

        .stat-grid, .fixture-grid, .cluster-grid, .bracket-grid {
          display: grid;
          gap: 14px;
        }

        .stat-grid {
          grid-template-columns: repeat(4, minmax(0, 1fr));
          margin-top: 24px;
        }

        .fixture-grid {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .cluster-grid {
          grid-template-columns: repeat(3, minmax(0, 1fr));
        }

        .bracket-grid {
          grid-template-columns: repeat(3, minmax(0, 1fr));
          margin-top: 12px;
        }

        .card, .metric-card, .fixture-card, .cluster-card, .bracket-card, .lineup-card {
          border: 1px solid rgba(214, 168, 79, 0.18);
          border-radius: 8px;
          background: linear-gradient(180deg, rgba(16, 20, 31, 0.98), rgba(23, 28, 42, 0.95));
          box-shadow: 0 18px 42px rgba(0, 0, 0, 0.24);
        }

        .card, .cluster-card, .bracket-card {
          padding: 20px;
        }

        .metric-card {
          padding: 16px;
        }

        .metric-label, .card-kicker {
          color: var(--muted-2);
          font-size: 0.78rem;
          font-weight: 800;
          letter-spacing: 0.08em;
          text-transform: uppercase;
        }

        .metric-value {
          color: var(--text);
          font-family: "Avenir Next Condensed", "DIN Condensed", sans-serif;
          font-size: 2rem;
          font-weight: 900;
          line-height: 1.05;
          margin-top: 4px;
        }

        .section-title {
          color: var(--text);
          font-family: "Avenir Next Condensed", "DIN Condensed", sans-serif;
          font-size: 2rem;
          font-weight: 900;
          text-transform: uppercase;
          margin: 1.4rem 0 0.3rem;
        }

        .section-copy {
          color: var(--muted);
          margin-bottom: 1rem;
        }

        .vs-badge {
          margin-top: 4.2rem;
          display: flex;
          align-items: center;
          justify-content: center;
          min-height: 58px;
          border: 1px solid rgba(244, 213, 141, 0.42);
          border-radius: 999px;
          color: var(--soft-gold);
          background: rgba(214, 168, 79, 0.08);
          font-weight: 900;
          font-size: 1.3rem;
        }

        .team-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 10px;
          margin-bottom: 10px;
        }

        .team-name {
          color: var(--text);
          font-family: "Avenir Next Condensed", "DIN Condensed", sans-serif;
          font-size: 1.7rem;
          font-weight: 900;
          text-transform: uppercase;
        }

        .pill {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          margin: 4px 5px 4px 0;
          padding: 5px 9px;
          border-radius: 999px;
          color: var(--text);
          background: rgba(255, 255, 255, 0.06);
          border: 1px solid rgba(255, 255, 255, 0.08);
          font-size: 0.78rem;
        }

        .gold {
          color: var(--soft-gold);
        }

        .prob-card {
          padding: 24px;
          border: 1px solid rgba(214, 168, 79, 0.26);
          border-radius: 8px;
          background:
            linear-gradient(135deg, rgba(16, 20, 31, 0.98), rgba(34, 41, 58, 0.92)),
            linear-gradient(90deg, rgba(214, 168, 79, 0.06), transparent);
        }

        .prob-row {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 12px;
          margin: 18px 0;
        }

        .prob-number {
          font-family: "Avenir Next Condensed", "DIN Condensed", sans-serif;
          font-size: clamp(2rem, 4vw, 3.5rem);
          font-weight: 900;
          line-height: 1;
        }

        .prob-label {
          color: var(--muted);
          font-size: 0.86rem;
          margin-top: 5px;
        }

        .prob-track {
          display: flex;
          height: 18px;
          overflow: hidden;
          border-radius: 999px;
          background: rgba(255, 255, 255, 0.08);
        }

        .fixture-card {
          padding: 16px;
        }

        .fixture-date {
          color: var(--soft-gold);
          font-weight: 800;
          font-size: 0.82rem;
          text-transform: uppercase;
        }

        .fixture-teams {
          color: var(--text);
          font-size: 1.1rem;
          font-weight: 850;
          margin: 8px 0 4px;
        }

        .fixture-meta {
          color: var(--muted);
          font-size: 0.86rem;
        }

        .score-chip {
          float: right;
          color: var(--text);
          border: 1px solid rgba(255,255,255,0.14);
          border-radius: 999px;
          padding: 4px 10px;
          background: rgba(255,255,255,0.07);
          font-weight: 800;
        }

        .bracket-stage {
          color: var(--soft-gold);
          font-family: "Avenir Next Condensed", "DIN Condensed", sans-serif;
          font-size: 1.35rem;
          font-weight: 900;
          text-transform: uppercase;
        }

        .mini-match {
          margin-top: 12px;
          padding-top: 12px;
          border-top: 1px solid rgba(255,255,255,0.08);
        }

        .data-note {
          color: var(--muted-2);
          font-size: 0.82rem;
        }

        [data-testid="stTabs"] button {
          color: var(--muted);
        }

        [data-testid="stTabs"] button[aria-selected="true"] {
          color: var(--soft-gold);
        }

        div.stButton > button {
          width: 100%;
          border: 1px solid rgba(244, 213, 141, 0.45);
          background: linear-gradient(135deg, #D6A84F, #8B692A);
          color: #05070D;
          font-weight: 900;
          border-radius: 8px;
          min-height: 3rem;
        }

        div.stButton > button:disabled {
          opacity: 0.35;
          color: rgba(5, 7, 13, 0.7);
        }

        [data-testid="stMetric"] {
          border: 1px solid rgba(214, 168, 79, 0.16);
          border-radius: 8px;
          background: rgba(16, 20, 31, 0.7);
          padding: 16px;
        }

        @media (max-width: 900px) {
          .stat-grid, .fixture-grid, .cluster-grid, .bracket-grid, .prob-row {
            grid-template-columns: 1fr;
          }

          .hero {
            padding: 26px 20px;
          }

          .hero:after {
            font-size: 118px;
            right: 10px;
          }

          .vs-badge {
            margin-top: 0;
            min-height: 42px;
          }
        }

        /* Coinbase-inspired visual direction: quiet canvas, one blue accent, pill controls. */
        :root {
          --bg: #ffffff;
          --surface: #ffffff;
          --surface-2: #f7f7f7;
          --surface-3: #eef0f3;
          --surface-strong: #eef0f3;
          --text: #0a0b0d;
          --body: #5b616e;
          --muted: #7c828a;
          --muted-2: #a8acb3;
          --hairline: #dee1e6;
          --hairline-soft: #eef0f3;
          --primary: #0052ff;
          --primary-active: #003ecc;
          --primary-disabled: #a8b8cc;
          --surface-dark: #0a0b0d;
          --surface-dark-elevated: #16181c;
          --on-dark: #ffffff;
          --on-dark-soft: #a8acb3;
          --semantic-up: #05b169;
          --semantic-down: #cf202f;
          --gold: #0052ff;
          --soft-gold: #0052ff;
          --green: #05b169;
          --amber: #7c828a;
          --red: #cf202f;
          --blue: #0052ff;
        }

        html, body, [data-testid="stAppViewContainer"] {
          background: var(--bg);
          color: var(--text);
          overflow-x: hidden;
        }

        [data-testid="stHeader"] {
          background: rgba(255, 255, 255, 0.88);
          backdrop-filter: blur(12px);
        }

        [data-testid="stMainBlockContainer"] {
          max-width: 1200px;
          padding: 32px 32px 96px;
        }

        h1, h2, h3, p, label, span, div {
          font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
          letter-spacing: 0;
        }

        .data-disclaimer {
          display: flex;
          align-items: flex-start;
          gap: 12px;
          border: 1px solid rgba(0, 82, 255, 0.18);
          border-radius: 18px;
          padding: 14px 16px;
          margin-bottom: 18px;
          background: linear-gradient(180deg, #f8fbff, #ffffff);
          color: var(--body);
        }

        .data-disclaimer strong {
          color: var(--text);
          font-weight: 600;
        }

        .data-disclaimer span {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          flex: 0 0 auto;
          width: 28px;
          height: 28px;
          border-radius: 9999px;
          background: rgba(0, 82, 255, 0.1);
          color: var(--primary);
          font-weight: 700;
        }

        .hero {
          position: relative;
          overflow: hidden;
          border: 0;
          border-radius: 24px;
          padding: clamp(40px, 6vw, 72px);
          background: var(--surface-dark);
          color: var(--on-dark);
          box-shadow: none;
        }

        .hero:after {
          display: none;
        }

        .hero-layout {
          display: grid;
          grid-template-columns: minmax(0, 1fr) minmax(320px, 0.85fr);
          gap: 48px;
          align-items: center;
          position: relative;
          z-index: 2;
        }

        .gradual-blur {
          position: absolute;
          pointer-events: none;
          isolation: isolate;
          z-index: 3;
        }

        .gradual-blur-bottom {
          left: 0;
          right: 0;
          bottom: -1px;
          height: 6rem;
        }

        .gradual-blur-top {
          left: 0;
          right: 0;
          top: 0;
          height: 6rem;
        }

        .gradual-blur-inner {
          position: relative;
          width: 100%;
          height: 100%;
        }

        .gradual-blur-inner > div {
          position: absolute;
          inset: 0;
          opacity: 0.95;
        }

        .gradual-blur-bottom .gb-layer-1 {
          -webkit-backdrop-filter: blur(0.16rem);
          backdrop-filter: blur(0.16rem);
          -webkit-mask-image: linear-gradient(to bottom, transparent 0%, black 20%, black 40%, transparent 60%);
          mask-image: linear-gradient(to bottom, transparent 0%, black 20%, black 40%, transparent 60%);
        }

        .gradual-blur-bottom .gb-layer-2 {
          -webkit-backdrop-filter: blur(0.34rem);
          backdrop-filter: blur(0.34rem);
          -webkit-mask-image: linear-gradient(to bottom, transparent 20%, black 40%, black 60%, transparent 80%);
          mask-image: linear-gradient(to bottom, transparent 20%, black 40%, black 60%, transparent 80%);
        }

        .gradual-blur-bottom .gb-layer-3 {
          -webkit-backdrop-filter: blur(0.62rem);
          backdrop-filter: blur(0.62rem);
          -webkit-mask-image: linear-gradient(to bottom, transparent 40%, black 60%, black 80%, transparent 100%);
          mask-image: linear-gradient(to bottom, transparent 40%, black 60%, black 80%, transparent 100%);
        }

        .gradual-blur-bottom .gb-layer-4 {
          -webkit-backdrop-filter: blur(0.98rem);
          backdrop-filter: blur(0.98rem);
          -webkit-mask-image: linear-gradient(to bottom, transparent 58%, black 78%, black 100%);
          mask-image: linear-gradient(to bottom, transparent 58%, black 78%, black 100%);
        }

        .gradual-blur-bottom .gb-layer-5 {
          background:
            linear-gradient(to bottom, transparent 0%, rgba(10, 11, 13, 0.28) 54%, rgba(255, 255, 255, 0.16) 100%),
            radial-gradient(circle at 50% 100%, rgba(125, 249, 255, 0.16), transparent 52%);
        }

        .hero-gradual-blur {
          opacity: 1;
        }

        .hero-title {
          max-width: 720px;
          color: var(--on-dark);
          font-size: clamp(3rem, 7vw, 5rem);
          font-weight: 400;
          line-height: 1;
          letter-spacing: -0.025em;
          text-transform: none;
          margin: 14px 0 20px;
        }

        .hero-copy {
          max-width: 650px;
          color: var(--on-dark-soft);
          font-size: 1.05rem;
          line-height: 1.6;
        }

        .eyebrow {
          display: inline-flex;
          align-items: center;
          width: fit-content;
          border-radius: 100px;
          padding: 6px 12px;
          background: rgba(255, 255, 255, 0.08);
          color: var(--on-dark);
          font-size: 0.75rem;
          font-weight: 600;
          letter-spacing: 0;
          text-transform: none;
        }

        .hero-actions {
          display: flex;
          flex-wrap: wrap;
          gap: 12px;
          margin-top: 28px;
        }

        .hero-actions a.hero-cta,
        .hero-actions a.hero-cta:visited {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          min-height: 48px;
          padding: 0 22px;
          border-radius: 100px;
          background: var(--primary);
          color: #ffffff !important;
          font-weight: 600;
          text-decoration: none !important;
          cursor: pointer;
        }

        .hero-actions a.hero-cta:hover {
          background: var(--primary-active);
          color: #ffffff !important;
          text-decoration: none !important;
        }

        .hero-actions a.hero-secondary,
        .hero-actions a.hero-secondary:visited {
          background: var(--surface-dark-elevated);
          color: var(--on-dark) !important;
        }

        .hero-actions a.hero-secondary:hover {
          background: #232832;
          color: var(--on-dark) !important;
        }

        .mockup-stack {
          display: grid;
          gap: 16px;
        }

        .mockup-card {
          border-radius: 24px;
          padding: 24px;
          background: var(--surface-dark-elevated);
          border: 1px solid rgba(255, 255, 255, 0.08);
          color: var(--on-dark);
        }

        .mockup-card.secondary {
          background: #111318;
        }

        .model-output-card {
          border-color: rgba(125, 249, 255, 0.22);
          box-shadow: 0 0 38px rgba(125, 249, 255, 0.08);
        }

        .mockup-row {
          display: grid;
          grid-template-columns: minmax(0, 1fr) auto;
          align-items: center;
          gap: 16px;
          padding: 14px 0;
          border-top: 1px solid rgba(255,255,255,0.08);
        }

        .mockup-row:first-child {
          border-top: 0;
        }

        .mockup-value {
          font-family: "JetBrains Mono", "SFMono-Regular", Menlo, Consolas, monospace;
          color: var(--on-dark);
          font-weight: 500;
          white-space: nowrap;
        }

        .model-output-card .mockup-value {
          color: #7df9ff;
        }

        .stat-grid {
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 16px;
          margin-top: 0;
        }

        .fixture-grid, .cluster-grid, .bracket-grid {
          gap: 24px;
        }

        .card, .metric-card, .fixture-card, .cluster-card, .bracket-card, .lineup-card, .prob-card {
          border: 1px solid var(--hairline);
          border-radius: 24px;
          background: var(--surface);
          box-shadow: none;
          color: var(--text);
        }

        .card, .cluster-card, .bracket-card, .prob-card {
          padding: 32px;
        }

        .metric-card, .fixture-card {
          padding: 24px;
        }

        @property --electric-angle {
          syntax: "<angle>";
          inherits: false;
          initial-value: 0deg;
        }

        .electric-border {
          --electric-border-color: #7df9ff;
          --electric-angle: 0deg;
          position: relative;
          isolation: isolate;
          border-radius: 24px;
          padding: 1px;
          margin: 6px 0 14px;
        }

        .electric-border::before {
          content: "";
          position: absolute;
          inset: -1px;
          z-index: 0;
          border-radius: inherit;
          padding: 1.5px;
          background:
            conic-gradient(
              from var(--electric-angle),
              transparent 0deg,
              transparent 38deg,
              color-mix(in srgb, var(--electric-border-color) 88%, white) 62deg,
              rgba(0, 82, 255, 0.72) 86deg,
              transparent 126deg,
              transparent 210deg,
              color-mix(in srgb, var(--electric-border-color) 70%, white) 240deg,
              transparent 292deg,
              transparent 360deg
            );
          -webkit-mask:
            linear-gradient(#000 0 0) content-box,
            linear-gradient(#000 0 0);
          -webkit-mask-composite: xor;
          mask-composite: exclude;
          animation: electric-angle 5.5s linear infinite;
          pointer-events: none;
        }

        .electric-border::after {
          content: "";
          position: absolute;
          inset: -10px;
          z-index: -1;
          border-radius: inherit;
          background:
            linear-gradient(120deg, color-mix(in srgb, var(--electric-border-color) 26%, transparent), transparent 42%),
            linear-gradient(-30deg, rgba(0, 82, 255, 0.22), transparent 58%);
          filter: blur(24px);
          opacity: 0.34;
          animation: electric-pulse 3.2s ease-in-out infinite alternate;
          pointer-events: none;
        }

        .electric-content {
          position: relative;
          z-index: 1;
          border-radius: inherit;
        }

        .electric-content > .card,
        .electric-content > .prob-card,
        .electric-content > .match-stage {
          border-color: color-mix(in srgb, var(--electric-border-color) 28%, var(--hairline));
          box-shadow:
            0 1px 0 rgba(255, 255, 255, 0.8) inset,
            0 22px 48px rgba(10, 11, 13, 0.08);
        }

        .team-electric .card {
          min-height: 172px;
        }

        .team-electric .team-name {
          font-size: clamp(1.3rem, 2vw, 1.65rem);
        }

        .match-stage {
          display: grid;
          grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
          gap: 20px;
          align-items: center;
          padding: 24px 28px;
          border: 1px solid var(--hairline);
          border-radius: 24px;
          background:
            linear-gradient(135deg, #ffffff, #f8fbff 46%, #ffffff),
            radial-gradient(circle at 50% 0%, rgba(125, 249, 255, 0.16), transparent 36%);
        }

        .match-stage-electric .pill {
          margin: 10px 0 0;
          background: rgba(0, 82, 255, 0.08);
          color: var(--primary);
        }

        .prediction-electric .prob-card {
          background:
            linear-gradient(180deg, #ffffff 0%, #f8fbff 100%),
            radial-gradient(circle at 50% 0%, rgba(125, 249, 255, 0.18), transparent 40%);
        }

        .insight-electric .card {
          background: linear-gradient(180deg, #ffffff, #fbfcff);
        }

        .explain-grid {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 16px;
          margin: 18px 0;
        }

        .explain-card {
          border: 1px solid var(--hairline);
          border-radius: 24px;
          padding: 22px;
          background: #ffffff;
        }

        .explain-card strong {
          display: block;
          color: var(--text);
          font-size: 1rem;
          font-weight: 600;
          margin-bottom: 8px;
        }

        .explain-card span {
          color: var(--body);
          font-size: 0.95rem;
          line-height: 1.5;
        }

        .kmeans-note {
          border: 1px solid rgba(0, 82, 255, 0.18);
          border-radius: 24px;
          padding: 24px;
          margin: 18px 0;
          background: linear-gradient(180deg, #f8fbff, #ffffff);
        }

        .kmeans-badge {
          display: inline-flex;
          align-items: center;
          width: fit-content;
          border-radius: 9999px;
          padding: 6px 12px;
          margin-bottom: 10px;
          background: rgba(0, 82, 255, 0.08);
          color: var(--primary);
          font-size: 0.82rem;
          font-weight: 600;
        }

        .match-side {
          min-width: 0;
        }

        .match-side-right {
          text-align: right;
        }

        .match-label {
          display: block;
          color: var(--muted);
          font-size: 0.78rem;
          font-weight: 600;
          margin-bottom: 4px;
        }

        .match-team {
          color: var(--text);
          font-size: clamp(1.35rem, 3vw, 2rem);
          font-weight: 600;
          line-height: 1.08;
          overflow-wrap: anywhere;
        }

        .match-count {
          display: block;
          margin-top: 8px;
          color: var(--body);
          font-family: "JetBrains Mono", "SFMono-Regular", Menlo, Consolas, monospace;
          font-size: 0.84rem;
        }

        .match-versus {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 54px;
          height: 54px;
          border-radius: 9999px;
          background: #0a0b0d;
          color: #ffffff;
          box-shadow:
            0 0 0 1px rgba(125, 249, 255, 0.42),
            0 0 28px rgba(125, 249, 255, 0.28);
          font-weight: 700;
        }

        @keyframes electric-angle {
          to {
            --electric-angle: 360deg;
          }
        }

        @keyframes electric-pulse {
          from {
            opacity: 0.18;
          }
          to {
            opacity: 0.46;
          }
        }

        .metric-label, .card-kicker {
          color: var(--muted);
          font-size: 0.78rem;
          font-weight: 600;
          letter-spacing: 0;
          text-transform: none;
        }

        .metric-value, .prob-number {
          color: var(--text);
          font-family: "JetBrains Mono", "SFMono-Regular", Menlo, Consolas, monospace;
          font-weight: 500;
          letter-spacing: 0;
        }

        .metric-value {
          font-size: 2rem;
          line-height: 1.2;
        }

        .section-title {
          color: var(--text);
          font-size: clamp(2.2rem, 4vw, 3.25rem);
          font-weight: 400;
          line-height: 1.08;
          letter-spacing: -0.02em;
          text-transform: none;
          margin: 96px 0 16px;
        }

        .section-copy {
          color: var(--body);
          font-size: 1rem;
          line-height: 1.6;
        }

        .vs-badge {
          margin-top: 3.8rem;
          min-height: 48px;
          border: 1px solid var(--hairline);
          border-radius: 9999px;
          color: var(--primary);
          background: var(--surface-strong);
          font-weight: 600;
          font-size: 1rem;
        }

        .team-name, .bracket-stage {
          color: var(--text);
          font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
          font-size: 1.25rem;
          font-weight: 600;
          text-transform: none;
          letter-spacing: 0;
        }

        .pill {
          border-radius: 9999px;
          color: var(--text);
          background: var(--surface-strong);
          border: 0;
          padding: 7px 12px;
          font-size: 0.82rem;
        }

        .gold, .fixture-date {
          color: var(--primary);
        }

        .prob-row {
          margin: 24px 0;
        }

        .prob-number {
          font-size: clamp(1.8rem, 3vw, 2.4rem);
        }

        .prob-label, .fixture-meta, .data-note {
          color: var(--body);
        }

        .prob-track {
          height: 10px;
          background: var(--surface-strong);
        }

        .fixture-date {
          font-weight: 600;
          font-size: 0.86rem;
          text-transform: none;
        }

        .fixture-teams {
          color: var(--text);
          font-size: 1rem;
          font-weight: 600;
        }

        .score-chip {
          color: var(--text);
          border: 0;
          background: var(--surface-strong);
          font-family: "JetBrains Mono", "SFMono-Regular", Menlo, Consolas, monospace;
          font-weight: 500;
        }

        .mini-match {
          border-top: 1px solid var(--hairline-soft);
        }

        [data-testid="stTabs"] {
          margin-top: 32px;
        }

        [data-testid="stTabs"] button {
          color: var(--body);
          font-weight: 500;
        }

        [data-testid="stTabs"] button[aria-selected="true"] {
          color: var(--primary);
        }

        [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
          background: var(--primary);
        }

        [data-testid="stPills"] {
          margin-top: 32px;
        }

        div.stButton > button {
          width: 100%;
          border: 0;
          background:
            linear-gradient(135deg, var(--primary), var(--primary-active) 72%, #00c8ff);
          color: #ffffff;
          font-weight: 600;
          border-radius: 100px;
          min-height: 48px;
          box-shadow:
            0 10px 24px rgba(0, 82, 255, 0.22),
            0 0 0 1px rgba(125, 249, 255, 0.32) inset;
        }

        div.stButton > button:hover {
          background:
            linear-gradient(135deg, var(--primary-active), #0034aa 72%, #00b7e8);
          color: #ffffff;
          box-shadow:
            0 14px 34px rgba(0, 82, 255, 0.28),
            0 0 0 4px rgba(125, 249, 255, 0.16);
        }

        div.stButton > button:disabled {
          background: var(--primary-disabled);
          color: #ffffff;
          opacity: 1;
          box-shadow: none;
        }

        .stSelectbox div[data-baseweb="select"],
        .stMultiSelect div[data-baseweb="select"],
        .stDateInput input {
          border-radius: 12px;
        }

        .stSelectbox div[data-baseweb="select"]:focus-within,
        .stMultiSelect div[data-baseweb="select"]:focus-within {
          box-shadow:
            0 0 0 1px rgba(125, 249, 255, 0.8),
            0 0 0 5px rgba(125, 249, 255, 0.14);
        }

        [data-testid="stDataFrame"], [data-testid="stTable"] {
          border-radius: 24px;
          overflow: hidden;
        }

        @media (max-width: 900px) {
          [data-testid="stMainBlockContainer"] {
            padding: 20px 16px 72px;
          }

          .hero-layout {
            grid-template-columns: 1fr;
          }

          .mockup-stack {
            min-height: auto;
          }

          .mockup-card.secondary {
            position: relative;
            width: 100%;
            margin-top: 16px;
            transform: none;
          }

          .stat-grid, .fixture-grid, .cluster-grid, .bracket-grid, .prob-row {
            grid-template-columns: 1fr;
          }

          .explain-grid {
            grid-template-columns: 1fr;
          }

          .match-stage {
            grid-template-columns: 1fr;
            text-align: left;
          }

          .match-side-right {
            text-align: left;
          }

          .match-versus {
            width: 46px;
            height: 46px;
          }

          .hero {
            padding: 32px 24px;
          }
        }

        @media (prefers-reduced-motion: reduce) {
          .electric-border::before,
          .electric-border::after {
            animation: none;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def fmt_number(value: float | int | None, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "-"
    if decimals == 0:
        return f"{value:,.0f}"
    return f"{value:,.{decimals}f}"


def pct(value: float) -> str:
    return f"{value:.1%}"


def html_block(markup: str) -> str:
    """Remove indentation that Markdown may otherwise interpret as code."""
    return "\n".join(line.strip() for line in markup.splitlines() if line.strip())


def electric_border(markup: str, class_name: str = "", color: str = "#7df9ff") -> str:
    return html_block(
        f"""
        <div class="electric-border {class_name}" style="--electric-border-color: {color};">
          <div class="electric-content">
            {markup}
          </div>
        </div>
        """
    )


def gradual_blur(position: str = "bottom", class_name: str = "") -> str:
    layers = "".join(f'<div class="gb-layer-{index}"></div>' for index in range(1, 6))
    return html_block(
        f"""
        <div class="gradual-blur gradual-blur-{position} {class_name}">
          <div class="gradual-blur-inner">{layers}</div>
        </div>
        """
    )


def render_data_disclaimer() -> None:
    st.markdown(
        """
        <div class="data-disclaimer">
          <span>i</span>
          <div>
            <strong>Dataset disclaimer:</strong>
            This project uses a non-official Kaggle dataset for learning and demonstration.
            The player data, fixtures, and model predictions should not be treated as real FIFA records, live results, betting odds, or official forecasts.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def active_view_from_query() -> str:
    requested = st.query_params.get(VIEW_QUERY_PARAM, VIEW_OPTIONS[0])
    if isinstance(requested, list):
        requested = requested[0] if requested else VIEW_OPTIONS[0]
    return requested if requested in VIEW_OPTIONS else VIEW_OPTIONS[0]


def humanize_feature(feature: str) -> str:
    cleaned = (
        feature.replace("total_", "")
        .replace("avg_", "average_")
        .replace("expected_goals_xg", "xG")
        .replace("expected_assists_xa", "xA")
        .replace("_", " ")
    )
    return cleaned.title().replace("Xg", "xG").replace("Xa", "xA")


def top_default_players(players: pd.DataFrame) -> list[str]:
    sort_cols = [
        col
        for col in ["matches_in_dataset", "minutes_played", "player_rating"]
        if col in players.columns
    ]
    if sort_cols:
        players = players.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    return players["player_name"].drop_duplicates().head(11).tolist()


def jersey_label(value: Any) -> str:
    if pd.isna(value):
        return "--"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(number)) if number.is_integer() else str(value)


def player_meta_lookup(players: pd.DataFrame) -> dict[str, dict[str, Any]]:
    columns = [col for col in ["player_name", "position", "jersey_number"] if col in players.columns]
    if "player_name" not in columns:
        return {}
    return (
        players[columns]
        .drop_duplicates("player_name")
        .set_index("player_name")
        .to_dict(orient="index")
    )


def player_display_name(player_name: str, lookup: dict[str, dict[str, Any]]) -> str:
    meta = lookup.get(player_name, {})
    jersey = jersey_label(meta.get("jersey_number"))
    position = meta.get("position", "")
    position_text = f" - {position}" if pd.notna(position) and str(position) else ""
    return f"#{jersey} {player_name}{position_text}"


def lineup_summary(players: pd.DataFrame, selected: list[str]) -> dict[str, float]:
    chosen = players[players["player_name"].isin(selected)].drop_duplicates("player_name")
    return {
        "count": float(len(chosen)),
        "rating": float(chosen["player_rating"].mean()) if "player_rating" in chosen else np.nan,
        "goals": float(chosen["goals"].sum()) if "goals" in chosen else np.nan,
        "xg": float(chosen["expected_goals_xg"].sum()) if "expected_goals_xg" in chosen else np.nan,
    }


def lineup_table(team: str, selected_players: list[str], profiles: pd.DataFrame, clustered: pd.DataFrame) -> pd.DataFrame:
    table = profiles[
        (profiles["team"] == team) & (profiles["player_name"].isin(selected_players))
    ].copy()
    keep_cols = [
        col
        for col in [
            "team",
            "jersey_number",
            "player_name",
            "position",
            "player_rating",
            "goals",
            "assists",
            "expected_goals_xg",
        ]
        if col in table.columns
    ]
    table = table[keep_cols].drop_duplicates(subset=["player_name"])
    sort_cols = [col for col in ["position", "jersey_number", "player_name"] if col in table.columns]
    if sort_cols:
        table = table.sort_values(sort_cols)

    if not clustered.empty and {"team", "player_name", "cluster_label"}.issubset(clustered.columns):
        cluster_cols = ["team", "player_name", "cluster_label"]
        table = table.merge(
            clustered[cluster_cols].drop_duplicates(),
            on=["team", "player_name"],
            how="left",
        )
    return table


def render_hero(profiles: pd.DataFrame, clustered: pd.DataFrame, match_count: int) -> None:
    team_count = profiles["team"].nunique()
    player_count = profiles["player_name"].nunique()
    cluster_count = clustered["cluster_label"].nunique() if not clustered.empty else 0

    st.markdown(
        html_block(
            f"""
        <div class="hero">
          <div class="hero-layout">
            <div>
              <div class="eyebrow">Football intelligence platform</div>
              <div class="hero-title">Lineup prediction for tournament analysis.</div>
              <div class="hero-copy">
                Compare selected elevens, inspect player profiles, and review player clusters
                in a calmer analytics workspace built around model outputs rather than spectacle.
              </div>
              <div class="hero-actions">
                <a class="hero-cta" href="?view=Match%20Predictor#view-content">Start prediction</a>
                <a class="hero-cta hero-secondary" href="?view=Player%20Explorer#view-content">Explore players</a>
              </div>
            </div>
            <div class="mockup-stack">
              <div class="mockup-card">
                <div class="card-kicker">Dataset coverage</div>
                <div class="mockup-row"><span>Teams</span><span class="mockup-value">{team_count}</span></div>
                <div class="mockup-row"><span>Players</span><span class="mockup-value">{player_count:,}</span></div>
                <div class="mockup-row"><span>Matches</span><span class="mockup-value">{match_count:,}</span></div>
                <div class="mockup-row"><span>Clusters</span><span class="mockup-value">{cluster_count}</span></div>
              </div>
              <div class="mockup-card secondary model-output-card">
                <div class="card-kicker">Model output</div>
                <div class="mockup-row"><span>Outcome probabilities</span><span class="mockup-value">W / D / L</span></div>
                <div class="mockup-row"><span>Lineup comparison</span><span class="mockup-value">63 signals</span></div>
                <div class="mockup-row"><span>Player profiles</span><span class="mockup-value">5 clusters</span></div>
              </div>
            </div>
          </div>
          {gradual_blur("bottom", "hero-gradual-blur")}
        </div>
        """
        ),
        unsafe_allow_html=True,
    )


def render_team_panel(team: str, selected_players: list[str], pool: pd.DataFrame, color: str) -> None:
    summary = lineup_summary(pool, selected_players)
    status = "Ready" if summary["count"] == 11 else "Needs 11"
    st.markdown(
        electric_border(
            f"""
        <div class="card">
          <div class="team-header">
            <div>
              <div class="card-kicker">Selected side</div>
              <div class="team-name">{escape(team)}</div>
            </div>
            <span class="pill">{int(summary["count"])} / 11 {status}</span>
          </div>
          <span class="pill">Avg rating {fmt_number(summary["rating"])}</span>
          <span class="pill">Goals {fmt_number(summary["goals"], 1)}</span>
          <span class="pill">xG {fmt_number(summary["xg"], 2)}</span>
        </div>
        """,
            class_name="team-electric",
            color=color,
        ),
        unsafe_allow_html=True,
    )


def render_lineup_pills(team: str, selected_players: list[str], profiles: pd.DataFrame, clustered: pd.DataFrame) -> None:
    table = lineup_table(team, selected_players, profiles, clustered)
    pills = []
    sort_cols = [col for col in ["position", "jersey_number", "player_name"] if col in table.columns]
    view = table.sort_values(sort_cols) if sort_cols else table
    for _, row in view.iterrows():
        label = escape(str(row["player_name"]))
        jersey = escape(jersey_label(row.get("jersey_number")))
        position = escape(str(row.get("position", "")))
        rating = fmt_number(row.get("player_rating"), 1)
        cluster = escape(str(row.get("cluster_label", ""))) if "cluster_label" in row else ""
        cluster_text = f" | {cluster}" if cluster and cluster != "nan" else ""
        pills.append(f'<span class="pill">#{jersey} {label} - {position} - rating {rating}{cluster_text}</span>')
    st.markdown("".join(pills) if pills else '<span class="pill">No players selected</span>', unsafe_allow_html=True)


def render_match_status(team_a: str, team_b: str, team_a_count: int, team_b_count: int, ready: bool) -> None:
    status = "Ready to predict" if ready else "Lineups pending"
    st.markdown(
        electric_border(
            f"""
        <div class="match-stage">
          <div class="match-side">
            <span class="match-label">Team A</span>
            <div class="match-team">{escape(team_a)}</div>
            <span class="match-count">{team_a_count}/11 selected</span>
          </div>
          <div class="match-versus">VS</div>
          <div class="match-side match-side-right">
            <span class="match-label">Team B</span>
            <div class="match-team">{escape(team_b)}</div>
            <span class="match-count">{team_b_count}/11 selected</span>
          </div>
        </div>
        <span class="pill">{status}</span>
        """,
            class_name="match-stage-electric",
            color="#7df9ff" if ready else "#0052ff",
        ),
        unsafe_allow_html=True,
    )


def build_probability_card(team_a: str, team_b: str, prediction: dict[str, Any]) -> str:
    a = prediction["team_a_win_probability"]
    d = prediction["draw_probability"]
    b = prediction["team_b_win_probability"]
    max_value = max(a, d, b)
    winner_label = team_a if max_value == a else "Draw" if max_value == d else team_b

    return electric_border(
        f"""
    <div class="prob-card">
      <div class="eyebrow">Model prediction</div>
      <div class="section-title" style="margin-top: 0.35rem;">{escape(team_a)} <span class="gold">VS</span> {escape(team_b)}</div>
      <div class="prob-row">
        <div><div class="prob-number" style="color: var(--primary);">{pct(a)}</div><div class="prob-label">{escape(team_a)} win</div></div>
        <div><div class="prob-number" style="color: var(--muted);">{pct(d)}</div><div class="prob-label">Draw</div></div>
        <div><div class="prob-number" style="color: var(--text);">{pct(b)}</div><div class="prob-label">{escape(team_b)} win</div></div>
      </div>
      <div class="prob-track">
        <div style="width: {a * 100:.2f}%; background: var(--primary);"></div>
        <div style="width: {d * 100:.2f}%; background: var(--surface-strong);"></div>
        <div style="width: {b * 100:.2f}%; background: var(--text);"></div>
      </div>
      <p class="section-copy" style="margin-bottom: 0; margin-top: 14px;">
        Highest estimated outcome: <span class="gold">{escape(winner_label)}</span>. These are probabilities, not guarantees.
      </p>
    </div>
    """,
        class_name="prediction-electric",
        color="#7df9ff",
    )


def matchup_difference_table(matchup_features: dict[str, float]) -> pd.DataFrame:
    focus_features = [
        "avg_player_rating",
        "total_expected_goals_xg",
        "total_expected_assists_xa",
        "total_goals",
        "total_assists",
        "total_creativity_score",
        "total_defensive_contribution",
        "avg_stamina_score",
        "avg_possession_impact",
        "total_saves",
    ]
    rows = []
    for feature in focus_features:
        a_key, b_key, d_key = f"team_a_{feature}", f"team_b_{feature}", f"diff_{feature}"
        if d_key not in matchup_features:
            continue
        rows.append(
            {
                "Metric": humanize_feature(feature),
                "Team A": matchup_features.get(a_key, 0.0),
                "Team B": matchup_features.get(b_key, 0.0),
                "Difference": matchup_features.get(d_key, 0.0),
            }
        )
    return pd.DataFrame(rows)


def explain_prediction(team_a: str, team_b: str, matchup_features: dict[str, float]) -> str:
    diff_rows = [
        (key.replace("diff_", ""), value)
        for key, value in matchup_features.items()
        if key.startswith("diff_")
    ]
    diff_rows = sorted(diff_rows, key=lambda item: abs(item[1]), reverse=True)
    useful = [item for item in diff_rows if abs(item[1]) > 0][:3]
    if not useful:
        return "The selected lineups are very close across the comparison features."

    phrases = []
    for feature, value in useful:
        leader = team_a if value > 0 else team_b
        phrases.append(f"{escape(leader)} leads in {escape(humanize_feature(feature).lower())}")
    return "The biggest lineup gaps are: " + ", ".join(phrases) + "."


def show_match_prediction(profiles: pd.DataFrame, clustered: pd.DataFrame, model, training_columns: list[str]) -> None:
    st.markdown('<div class="section-title">Match Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Choose two teams, lock in 11 players per side, then generate estimated outcome probabilities.</p>',
        unsafe_allow_html=True,
    )

    teams = sorted(profiles["team"].dropna().unique())
    select_a, vs_col, select_b = st.columns([5, 1, 5])
    with select_a:
        team_a = st.selectbox("Team A", teams, index=0)
    with vs_col:
        st.markdown('<div class="vs-badge">VS</div>', unsafe_allow_html=True)
    with select_b:
        default_b = 1 if len(teams) > 1 else 0
        team_b = st.selectbox("Team B", teams, index=default_b)

    team_a_pool = profiles[profiles["team"] == team_a].drop_duplicates(subset=["player_name"])
    team_b_pool = profiles[profiles["team"] == team_b].drop_duplicates(subset=["player_name"])
    team_a_options = sorted(team_a_pool["player_name"].tolist())
    team_b_options = sorted(team_b_pool["player_name"].tolist())
    team_a_lookup = player_meta_lookup(team_a_pool)
    team_b_lookup = player_meta_lookup(team_b_pool)
    team_a_key = f"team_a_players_{team_a}"
    team_b_key = f"team_b_players_{team_b}"

    col_a, col_b = st.columns(2)
    with col_a:
        render_team_panel(
            team_a,
            st.session_state.get(team_a_key, top_default_players(team_a_pool)),
            team_a_pool,
            "#7df9ff",
        )
        team_a_players = st.multiselect(
            f"{team_a} lineup",
            team_a_options,
            default=top_default_players(team_a_pool),
            format_func=lambda player: player_display_name(player, team_a_lookup),
            key=team_a_key,
        )
        render_lineup_pills(team_a, team_a_players, profiles, clustered)

    with col_b:
        render_team_panel(
            team_b,
            st.session_state.get(team_b_key, top_default_players(team_b_pool)),
            team_b_pool,
            "#0052ff",
        )
        team_b_players = st.multiselect(
            f"{team_b} lineup",
            team_b_options,
            default=top_default_players(team_b_pool),
            format_func=lambda player: player_display_name(player, team_b_lookup),
            key=team_b_key,
        )
        render_lineup_pills(team_b, team_b_players, profiles, clustered)

    ready = len(team_a_players) == 11 and len(team_b_players) == 11 and team_a != team_b
    render_match_status(team_a, team_b, len(team_a_players), len(team_b_players), ready)

    if team_a == team_b:
        st.warning("Choose two different teams.")
    elif len(team_a_players) != 11 or len(team_b_players) != 11:
        st.warning("Each team needs exactly 11 selected players.")

    if st.button("Predict Match", type="primary", disabled=not ready):
        try:
            prediction = predict_custom_match(
                team_a,
                team_b,
                team_a_players,
                team_b_players,
                profiles,
                model,
                training_columns,
            )
        except Exception:
            LOGGER.exception("Match prediction failed")
            st.error("The prediction could not be generated. Please adjust the selected lineups and try again.")
            return

        st.markdown(build_probability_card(team_a, team_b, prediction), unsafe_allow_html=True)
        st.markdown(
            electric_border(
                f'<div class="card"><div class="card-kicker">Why this prediction?</div><p class="section-copy" style="margin: 8px 0 0;">{explain_prediction(team_a, team_b, prediction["matchup_features"])}</p></div>',
                class_name="insight-electric",
                color="#0052ff",
            ),
            unsafe_allow_html=True,
        )

        diff_df = matchup_difference_table(prediction["matchup_features"])
        if not diff_df.empty:
            st.markdown('<div class="section-title">Matchup Insights</div>', unsafe_allow_html=True)
            st.dataframe(
                diff_df.style.format({"Team A": "{:.2f}", "Team B": "{:.2f}", "Difference": "{:+.2f}"}),
                width="stretch",
                hide_index=True,
            )

        st.markdown('<div class="section-title">Selected Lineups</div>', unsafe_allow_html=True)
        left, right = st.columns(2)
        left.dataframe(lineup_table(team_a, team_a_players, profiles, clustered), width="stretch", hide_index=True)
        right.dataframe(lineup_table(team_b, team_b_players, profiles, clustered), width="stretch", hide_index=True)


def cluster_description(label: str) -> str:
    descriptions = {
        "Efficient Finishers": "Strong conversion, positive goal output, and useful attacking impact.",
        "Creative Playmakers": "High creativity, key passes, assists, and possession impact.",
        "Underperforming Attackers": "Chance volume is present, but output trails expected contribution.",
        "Defensive Specialists": "Tackles, interceptions, clearances, saves, and defensive contribution stand out.",
        "Low-Impact / Low-Minute Players": "Lower minutes and lower broad contribution in the aggregated profiles.",
    }
    return descriptions.get(label, "Players grouped by similar performance statistics.")


def cluster_cards(clustered: pd.DataFrame) -> str:
    cards = []
    for label, group in clustered.groupby("cluster_label", sort=True):
        top_players = (
            group.sort_values("player_rating", ascending=False)["player_name"]
            .drop_duplicates()
            .head(3)
            .tolist()
        )
        top_text = ", ".join(escape(str(player)) for player in top_players)
        avg_rating = group["player_rating"].mean() if "player_rating" in group else np.nan
        cards.append(
            html_block(
                f"""
            <div class="cluster-card">
              <div class="card-kicker">{len(group):,} players</div>
              <div class="bracket-stage">{escape(str(label))}</div>
              <p class="section-copy" style="margin: 8px 0;">{escape(cluster_description(str(label)))}</p>
              <span class="pill">Avg rating {fmt_number(avg_rating)}</span>
              <span class="pill">Top: {top_text}</span>
            </div>
            """
            )
        )
    return html_block(f'<div class="cluster-grid">{"".join(cards)}</div>')


def show_player_clusters(clustered: pd.DataFrame) -> None:
    if clustered.empty:
        st.info("Clustered player profiles are not available yet. Run python main.py first.")
        return

    st.markdown('<div class="section-title">Player Clusters</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Unsupervised learning groups players by similar output, style, and overperformance patterns.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(cluster_cards(clustered), unsafe_allow_html=True)

    cluster_labels = sorted(clustered["cluster_label"].dropna().unique())
    selected_label = st.selectbox("Cluster", ["All clusters"] + cluster_labels)
    view = clustered if selected_label == "All clusters" else clustered[clustered["cluster_label"] == selected_label]

    display_cols = [
        col
        for col in [
            "team",
            "player_name",
            "position",
            "cluster_label",
            "player_rating",
            "goals",
            "assists",
            "expected_goals_xg",
            "expected_assists_xa",
            "goal_overperformance",
            "assist_overperformance",
            "goal_contribution_overperformance",
        ]
        if col in view.columns
    ]
    st.dataframe(view[display_cols].sort_values(["cluster_label", "team", "player_name"]), width="stretch", hide_index=True)

    metric_cols = [
        col
        for col in [
            "player_rating",
            "minutes_played",
            "goals",
            "assists",
            "expected_goals_xg",
            "expected_assists_xa",
            "goal_overperformance",
            "assist_overperformance",
            "goal_contribution_overperformance",
            "shot_accuracy",
            "creativity_score",
            "offensive_contribution",
            "defensive_contribution",
        ]
        if col in clustered.columns
    ]
    if metric_cols:
        st.markdown('<div class="section-title">Average Stats by Cluster</div>', unsafe_allow_html=True)
        summary = clustered.groupby("cluster_label")[metric_cols].mean().round(3)
        st.dataframe(summary, width="stretch")

    image_cols = st.columns(2)
    pca_image = OUTPUTS_DIR / "figures" / "player_clusters_pca.png"
    heatmap_image = OUTPUTS_DIR / "figures" / "cluster_profile_heatmap.png"
    if pca_image.exists():
        image_cols[0].image(str(pca_image), caption="PCA cluster map", width="stretch")
    if heatmap_image.exists():
        image_cols[1].image(str(heatmap_image), caption="Cluster profile heatmap", width="stretch")


def show_player_explorer(clustered: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Player Explorer</div>', unsafe_allow_html=True)
    if clustered.empty:
        st.info("Clustered player profiles are not available yet. Run python main.py first.")
        return

    filters = st.columns([2, 2, 2, 2])
    with filters[0]:
        team = st.selectbox("Team", ["All teams"] + sorted(clustered["team"].dropna().unique()))
    with filters[1]:
        position = st.selectbox("Position", ["All positions"] + sorted(clustered["position"].dropna().unique()))
    with filters[2]:
        cluster = st.selectbox("Cluster label", ["All clusters"] + sorted(clustered["cluster_label"].dropna().unique()))
    with filters[3]:
        min_rating = st.slider("Minimum rating", 0.0, 10.0, 0.0, 0.1)

    view = clustered.copy()
    if team != "All teams":
        view = view[view["team"] == team]
    if position != "All positions":
        view = view[view["position"] == position]
    if cluster != "All clusters":
        view = view[view["cluster_label"] == cluster]
    if "player_rating" in view:
        view = view[view["player_rating"] >= min_rating]

    perf_mode = st.radio(
        "Overperformance view",
        ["All players", "Goal overperformers", "Goal underperformers"],
        horizontal=True,
    )
    if perf_mode == "Goal overperformers" and "goal_overperformance" in view:
        view = view[view["goal_overperformance"] > 0]
    elif perf_mode == "Goal underperformers" and "goal_overperformance" in view:
        view = view[view["goal_overperformance"] < 0]

    if view.empty:
        st.info("No players match those filters.")
        return

    player_options = view.sort_values(["team", "player_name"])["player_name"].drop_duplicates().tolist()
    selected_player = st.selectbox("Player detail", player_options)
    player = view[view["player_name"] == selected_player].sort_values("player_rating", ascending=False).iloc[0]

    st.markdown(
        f"""
        <div class="card">
          <div class="card-kicker">{escape(str(player["team"]))} - {escape(str(player["position"]))}</div>
          <div class="team-name">{escape(str(player["player_name"]))}</div>
          <span class="pill">Cluster: {escape(str(player.get("cluster_label", "-")))}</span>
          <span class="pill">Rating {fmt_number(player.get("player_rating"))}</span>
          <span class="pill">Goal overperformance {fmt_number(player.get("goal_overperformance"), 2)}</span>
          <span class="pill">Assist overperformance {fmt_number(player.get("assist_overperformance"), 2)}</span>
          <p class="section-copy" style="margin: 12px 0 0;">Positive overperformance means actual output is above expected output in the aggregated player profile.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    display_cols = [
        col
        for col in [
            "team",
            "player_name",
            "position",
            "cluster_label",
            "goals",
            "expected_goals_xg",
            "goal_overperformance",
            "assists",
            "expected_assists_xa",
            "assist_overperformance",
            "player_rating",
        ]
        if col in view.columns
    ]
    st.dataframe(view[display_cols].sort_values("player_rating", ascending=False), width="stretch", hide_index=True)


def selected_cluster_count(clustered: pd.DataFrame) -> int | None:
    if clustered.empty:
        return None
    if "cluster_label" in clustered.columns:
        return int(clustered["cluster_label"].dropna().nunique())
    if "cluster" in clustered.columns:
        return int(clustered["cluster"].dropna().nunique())
    return None


def render_kmeans_selection(k_path: Path, clustered: pd.DataFrame) -> None:
    kmeans = pd.read_csv(k_path)
    selected_k = selected_cluster_count(clustered)
    best_row = kmeans.loc[kmeans["silhouette_score"].idxmax()]
    best_k = int(best_row["k"])
    best_score = float(best_row["silhouette_score"])
    selected_row = kmeans[kmeans["k"] == selected_k] if selected_k is not None else pd.DataFrame()
    selected_score = float(selected_row["silhouette_score"].iloc[0]) if not selected_row.empty else np.nan

    st.markdown('<div class="section-title">K-Means Selection</div>', unsafe_allow_html=True)
    st.markdown(
        html_block(
            f"""
        <div class="kmeans-note">
          <div class="kmeans-badge">How to read this table</div>
          <p class="section-copy" style="margin: 0;">
            K-Means tries different numbers of player groups. A useful choice should keep similar players close together
            while still producing groups that are easy to explain in football terms. This project uses
            <strong>{selected_k if selected_k is not None else "the selected"} clusters</strong>
            for the player labels shown elsewhere in the app.
          </p>
        </div>
        <div class="explain-grid">
          <div class="explain-card"><strong>k</strong><span>The number of player groups being tested.</span></div>
          <div class="explain-card"><strong>Inertia</strong><span>Lower means players are tighter inside their groups, but it almost always drops as k gets bigger.</span></div>
          <div class="explain-card"><strong>Silhouette score</strong><span>Higher means groups are more separated. Best tested here: k={best_k} at {best_score:.3f}.</span></div>
        </div>
        """
        ),
        unsafe_allow_html=True,
    )

    if selected_k is not None and np.isfinite(selected_score):
        st.markdown(
            f"""
            <div class="card">
              <div class="card-kicker">Why the app uses k={selected_k}</div>
              <p class="section-copy" style="margin: 8px 0 0;">
                k={best_k} has the highest silhouette score, but k={selected_k} creates the five readable football roles used in the app:
                finishers, playmakers, underperforming attackers, defensive specialists, and low-minute players.
                Its silhouette score is {selected_score:.3f}, close enough to preserve separation while making the clusters easier to interpret.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    readable = kmeans.rename(
        columns={
            "k": "Number of clusters (k)",
            "inertia": "Inertia (lower = tighter)",
            "silhouette_score": "Silhouette score (higher = better)",
        }
    ).copy()
    if selected_k is not None:
        readable["App choice"] = np.where(
            readable["Number of clusters (k)"] == selected_k,
            "Used for player clusters",
            "Compared",
        )

    st.dataframe(
        readable.style.format(
            {
                "Inertia (lower = tighter)": "{:,.1f}",
                "Silhouette score (higher = better)": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )


def show_model_insights(clustered: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Model Insights</div>', unsafe_allow_html=True)
    metric_path = OUTPUTS_DIR / "supervised_model_metrics.csv"
    importance_path = OUTPUTS_DIR / "feature_importances.csv"
    k_path = OUTPUTS_DIR / "kmeans_k_selection.csv"

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            <div class="card">
              <div class="card-kicker">Supervised learning</div>
              <p class="section-copy">The match model learns from matchup-level rows. The target is W, D, or L from Team A's perspective, and the output is a probability distribution.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            """
            <div class="card">
              <div class="card-kicker">Unsupervised learning</div>
              <p class="section-copy">K-Means clusters players using scaled profile metrics and engineered overperformance features. Cluster names are interpretive labels.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if metric_path.exists():
        st.markdown('<div class="section-title">Supervised Model Comparison</div>', unsafe_allow_html=True)
        st.dataframe(pd.read_csv(metric_path), width="stretch", hide_index=True)

    if importance_path.exists():
        st.markdown('<div class="section-title">Top Model Factors</div>', unsafe_allow_html=True)
        importances = pd.read_csv(importance_path).head(15)
        st.bar_chart(importances.set_index("feature")["importance"])

    if k_path.exists():
        render_kmeans_selection(k_path, clustered)


def main() -> None:
    st.set_page_config(page_title="FIFA World Cup ML Analytics", page_icon="26", layout="wide")
    inject_css()

    try:
        model, training_columns = load_model_artifacts()
        profiles, clustered = load_player_data()
        match_count = load_match_count()
    except FileNotFoundError:
        st.error("Required artifacts are missing. Run python main.py before starting the app.")
        st.stop()
    except Exception:
        LOGGER.exception("Failed to load app artifacts")
        st.error("The app could not load its saved model artifacts. Please check the deployment files.")
        st.stop()

    active_view = active_view_from_query()
    render_data_disclaimer()
    render_hero(profiles, clustered, match_count)

    st.markdown('<div id="view-content"></div>', unsafe_allow_html=True)
    selected_view = st.pills(
        "View",
        VIEW_OPTIONS,
        default=active_view,
        key=f"view_nav_{active_view}",
        label_visibility="collapsed",
        width="stretch",
    )
    selected_view = selected_view or active_view

    if selected_view != active_view:
        st.query_params[VIEW_QUERY_PARAM] = selected_view
        st.rerun()

    if selected_view == "Match Predictor":
        show_match_prediction(profiles, clustered, model, training_columns)
    elif selected_view == "Player Clusters":
        show_player_clusters(clustered)
    elif selected_view == "Player Explorer":
        show_player_explorer(clustered)
    elif selected_view == "Model Insights":
        show_model_insights(clustered)


if __name__ == "__main__":
    main()
