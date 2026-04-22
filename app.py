import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────
# CSS — dark industrial theme
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0f14;
    color: #e8eaf0;
}
.stApp { background-color: #0d0f14; }
section[data-testid="stSidebar"] {
    background-color: #151820 !important;
    border-right: 1px solid #252836;
}
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

.metric-card {
    background: #151820;
    border: 1px solid #252836;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.75rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
}
.badge-churn {
    display: inline-block;
    background: rgba(255,77,109,0.15);
    border: 1px solid #ff4d6d;
    color: #ff4d6d;
    padding: 8px 20px;
    border-radius: 999px;
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    font-weight: 700;
}
.badge-safe {
    display: inline-block;
    background: rgba(0,245,160,0.12);
    border: 1px solid #00f5a0;
    color: #00f5a0;
    padding: 8px 20px;
    border-radius: 999px;
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    font-weight: 700;
}
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #00e5ff;
    border-bottom: 1px solid #252836;
    padding-bottom: 8px;
    margin: 28px 0 16px 0;
}
.gauge-track {
    background: #252836;
    border-radius: 999px;
    height: 10px;
    width: 100%;
    margin: 8px 0;
    overflow: hidden;
}
.gauge-fill { height: 100%; border-radius: 999px; }
hr { border-color: #252836; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.stButton > button {
    background: #00e5ff !important;
    color: #000 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    width: 100% !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# CHARGEMENT DES ARTEFACTS
# ─────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open('models/xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_names.json', 'r') as f:
        feature_names = json.load(f)
    return model, scaler, feature_names

try:
    model, scaler, feature_names = load_artifacts()
    model_loaded = True
except FileNotFoundError as e:
    model_loaded = False
    st.error(f"Fichiers manquants : {e}. Lance d'abord la cell de sauvegarde dans le notebook.")
    st.stop()

THRESHOLD = 0.40


# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 20px 0 10px 0;">
    <span style="font-family:Space Mono,monospace; font-size:0.7rem;
                 color:#00e5ff; text-transform:uppercase; letter-spacing:0.15em;">
        Customer Intelligence Platform
    </span>
    <h1 style="margin:4px 0 0 0; font-size:2rem; color:#e8eaf0;">
        Churn Prediction Dashboard
    </h1>
    <p style="color:#6b7280; margin-top:4px; font-size:0.9rem;">
        Modele XGBoost &nbsp;·&nbsp; Recall=80% &nbsp;·&nbsp; ROC-AUC=0.795 &nbsp;·&nbsp; Seuil=0.40
    </p>
</div>
<hr style="margin:0 0 24px 0;">
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# SIDEBAR — INPUT CLIENT
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:Space Mono,monospace; font-size:0.65rem;
                color:#00e5ff; text-transform:uppercase; letter-spacing:0.15em;
                margin-bottom:16px;">
        ◈ Profil Client
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Comportement**")
    age                  = st.slider("Age", 18, 74, 46)
    tenure_months        = st.slider("Anciennete (mois)", 1, 59, 30)
    monthly_logins       = st.slider("Connexions / mois", 0, 54, 20)
    weekly_active_days   = st.slider("Jours actifs / semaine", 0, 7, 3)
    avg_session_time     = st.slider("Duree session (min)", 1, 42, 15)
    last_login_days_ago  = st.slider("Derniere connexion (jours)", 0, 80, 9)
    features_used        = st.slider("Fonctionnalites utilisees", 1, 15, 5)
    usage_growth_rate    = st.slider("Taux de croissance usage", -0.6, 0.5, 0.0, step=0.05)

    st.markdown("---")
    st.markdown("**Financier**")
    monthly_fee          = st.slider("Abonnement mensuel (EUR)", 10, 100, 30)
    total_revenue        = st.slider("Revenu total (EUR)", 10, 5900, 720)
    payment_failures     = st.slider("Echecs de paiement", 0, 5, 0)

    st.markdown("---")
    st.markdown("**Support & Satisfaction**")
    support_tickets      = st.slider("Tickets support", 0, 7, 1)
    avg_resolution_time  = st.slider("Temps resolution (h)", 1, 62, 24)
    csat_score           = st.slider("Score satisfaction (CSAT 1-5)", 1, 5, 4)
    escalations          = st.slider("Escalades", 0, 4, 0)
    nps_score            = st.slider("NPS Score", -100, 100, 19)

    st.markdown("---")
    st.markdown("**Marketing**")
    email_open_rate      = st.slider("Taux ouverture email", 0.1, 0.9, 0.5, step=0.05)
    marketing_click_rate = st.slider("Taux clic marketing", 0.0, 0.5, 0.2, step=0.05)
    referral_count       = st.slider("Parrainages", 0, 7, 1)

    st.markdown("---")
    predict_btn = st.button("ANALYSER CE CLIENT")


# ─────────────────────────────────────────────────────────
# KPIs MODELE — toujours visibles
# ─────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Performance du modele</div>', unsafe_allow_html=True)

cols = st.columns(5)
kpis = [
    ("ROC-AUC", "0.795", "#00e5ff"),
    ("Recall Churn", "80%", "#00f5a0"),
    ("F1 Churn", "0.39", "#00e5ff"),
    ("Precision Churn", "26%", "#ffbe0b"),
    ("Churners detectes", "163/204", "#00f5a0"),
]
for col, (label, val, color) in zip(cols, kpis):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{color};">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────
def build_input():
    """Construit le vecteur d'entree a partir des sliders."""
    base = {
        'age': age,
        'tenure_months': tenure_months,
        'monthly_logins': monthly_logins,
        'weekly_active_days': weekly_active_days,
        'avg_session_time': avg_session_time,
        'features_used': features_used,
        'usage_growth_rate': usage_growth_rate,
        'last_login_days_ago': last_login_days_ago,
        'monthly_fee': monthly_fee,
        'total_revenue': total_revenue,
        'payment_failures': payment_failures,
        'support_tickets': support_tickets,
        'avg_resolution_time': avg_resolution_time,
        'csat_score': csat_score,
        'escalations': escalations,
        'email_open_rate': email_open_rate,
        'marketing_click_rate': marketing_click_rate,
        'nps_score': nps_score,
        'referral_count': referral_count,
    }
    df = pd.DataFrame([base])
    # Aligner sur les feature_names (colonnes encodees)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names]


if predict_btn:
    input_df = build_input()
    input_scaled = scaler.transform(input_df)
    proba = model.predict_proba(input_scaled)[0][1]
    prediction = int(proba >= THRESHOLD)

    # ── RESULTAT PRINCIPAL ──
    st.markdown('<div class="section-title">Resultat de l\'analyse</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 2])

    color_main = "#ff4d6d" if prediction == 1 else "#00f5a0"
    with c1:
        st.markdown(f"""
        <div class="metric-card" style="border-color:{color_main}33;">
            <div class="metric-value" style="color:{color_main}; font-size:2.8rem;">
                {proba:.0%}
            </div>
            <div class="metric-label">Probabilite de churn</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        badge = '<span class="badge-churn">CLIENT A RISQUE</span>' if prediction == 1 \
                else '<span class="badge-safe">CLIENT STABLE</span>'
        action = "Declencher une action de retention" if prediction == 1 \
                 else "Aucune action urgente requise"
        action_color = "#ff4d6d" if prediction == 1 else "#00f5a0"
        st.markdown(f"""
        <div class="metric-card" style="display:flex; flex-direction:column;
             justify-content:center; gap:12px;">
            {badge}
            <div style="font-size:0.8rem; color:{action_color}; margin-top:4px;">
                {action}
            </div>
            <div style="font-size:0.75rem; color:#6b7280;">
                Seuil applique : {THRESHOLD:.0%}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        pct = int(proba * 100)
        if proba < 0.35:
            gc, rl = "#00f5a0", "RISQUE FAIBLE"
        elif proba < 0.60:
            gc, rl = "#ffbe0b", "RISQUE MODERE"
        else:
            gc, rl = "#ff4d6d", "RISQUE ELEVE"

        st.markdown(f"""
        <div class="metric-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-family:Space Mono,monospace; font-size:0.7rem;
                             color:{gc}; letter-spacing:0.1em;">{rl}</span>
                <span style="font-family:Space Mono,monospace; font-size:1.1rem;
                             color:{gc};">{pct}%</span>
            </div>
            <div class="gauge-track" style="margin-top:12px;">
                <div class="gauge-fill" style="width:{pct}%; background:{gc};"></div>
            </div>
            <div style="display:flex; justify-content:space-between;
                        font-size:0.7rem; color:#6b7280; margin-top:4px;">
                <span>0%</span><span>Seuil 40%</span><span>100%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── SHAP INDIVIDUEL ──
    st.markdown('<div class="section-title">Explication de la prediction (SHAP)</div>',
                unsafe_allow_html=True)
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(input_scaled)

        shap_series = pd.Series(shap_vals[0], index=feature_names)
        top_features = shap_series.abs().nlargest(12).index[::-1]
        top_vals = shap_series[top_features]

        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor('#151820')
        ax.set_facecolor('#151820')
        colors = ['#ff4d6d' if v > 0 else '#00f5a0' for v in top_vals]
        ax.barh(range(12), top_vals.values, color=colors, edgecolor='none', height=0.6)
        ax.set_yticks(range(12))
        ax.set_yticklabels(top_features, color='#e8eaf0', fontsize=9, fontfamily='monospace')
        ax.axvline(0, color='#252836', linewidth=1.5)
        ax.set_xlabel("Valeur SHAP", color='#6b7280', fontsize=8)
        ax.tick_params(colors='#6b7280', labelsize=8)
        for s in ax.spines.values():
            s.set_edgecolor('#252836')
        red_p = mpatches.Patch(color='#ff4d6d', label='Pousse vers le churn')
        grn_p = mpatches.Patch(color='#00f5a0', label='Reduit le risque')
        ax.legend(handles=[red_p, grn_p], facecolor='#151820',
                  edgecolor='#252836', labelcolor='#e8eaf0', fontsize=8)
        ax.set_title("Impact individuel des features", color='#e8eaf0',
                     fontsize=10, fontfamily='monospace', pad=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    except ImportError:
        st.info("Installe shap pour voir les explications individuelles : pip install shap")
        importances = model.feature_importances_
        fi = pd.Series(importances, index=feature_names).nlargest(12)[::-1]
        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor('#151820')
        ax.set_facecolor('#151820')
        ax.barh(fi.index, fi.values, color='#00e5ff', height=0.6)
        ax.set_xlabel("Importance globale", color='#6b7280', fontsize=8)
        ax.tick_params(colors='#9ca3af')
        for s in ax.spines.values():
            s.set_edgecolor('#252836')
        ax.set_title("Feature Importance (XGBoost)", color='#e8eaf0',
                     fontsize=10, fontfamily='monospace')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── RECOMMANDATIONS BUSINESS ──
    st.markdown('<div class="section-title">Recommandations business</div>',
                unsafe_allow_html=True)
    recos = []
    if last_login_days_ago > 14:
        recos.append(("Inactivite detectee",
                      f"{last_login_days_ago} jours sans connexion → Email de reactivation immediat",
                      "#ff4d6d"))
    if payment_failures >= 2:
        recos.append(("Echecs de paiement",
                      f"{payment_failures} echecs → Contact service client proactif",
                      "#ff4d6d"))
    if csat_score <= 2:
        recos.append(("Satisfaction faible",
                      f"CSAT={csat_score}/5 → Offrir geste commercial ou upgrade",
                      "#ffbe0b"))
    if avg_resolution_time > 40:
        recos.append(("Support lent",
                      f"Resolution en {avg_resolution_time}h → Escalade prioritaire",
                      "#ffbe0b"))
    if monthly_logins < 8:
        recos.append(("Faible engagement",
                      f"Seulement {monthly_logins} connexions/mois → Campagne engagement",
                      "#ffbe0b"))
    if tenure_months < 6:
        recos.append(("Client recent",
                      f"Seulement {tenure_months} mois → Programme onboarding renforce",
                      "#00e5ff"))
    if not recos:
        recos.append(("Profil sain",
                      "Aucun signal d'alerte majeur — suivi standard",
                      "#00f5a0"))

    r1, r2 = st.columns(2)
    for i, (title, desc, color) in enumerate(recos):
        col = r1 if i % 2 == 0 else r2
        with col:
            st.markdown(f"""
            <div style="background:#151820; border:1px solid {color}33;
                        border-left:3px solid {color}; border-radius:8px;
                        padding:14px 16px; margin-bottom:10px;">
                <div style="font-size:0.9rem; margin-bottom:4px;">
                    <strong style="color:{color};">{title}</strong>
                </div>
                <div style="font-size:0.8rem; color:#9ca3af;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# ETAT INITIAL — Feature importance globale
# ─────────────────────────────────────────────────────────
else:
    st.markdown('<div class="section-title">Guide d\'utilisation</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#151820; border:1px solid #252836; border-radius:12px;
                padding:32px; text-align:center; color:#6b7280;">
        <div style="font-size:2.5rem; margin-bottom:12px;">📡</div>
        <div style="font-family:Space Mono,monospace; font-size:0.9rem;
                    color:#e8eaf0; margin-bottom:8px;">
            Configurez le profil client dans le panneau gauche
        </div>
        <div style="font-size:0.85rem;">
            Ajustez les curseurs puis cliquez sur
            <strong style="color:#00e5ff;">ANALYSER CE CLIENT</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Top 10 features globales — XGBoost</div>',
                unsafe_allow_html=True)
    fi_series = pd.Series(model.feature_importances_,
                          index=feature_names).nlargest(10)[::-1]
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#151820')
    ax.set_facecolor('#151820')
    colors_bar = ['#00e5ff' if i >= 7 else '#1e3a4a' for i in range(len(fi_series))]
    ax.barh(fi_series.index, fi_series.values, color=colors_bar, height=0.6)
    ax.set_xlabel("Importance", color='#6b7280', fontsize=9)
    ax.tick_params(colors='#9ca3af', labelsize=9)
    for s in ax.spines.values():
        s.set_edgecolor('#252836')
    ax.set_title("Feature Importance globale — XGBoost", color='#e8eaf0',
                 fontsize=11, fontfamily='monospace', pad=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────
st.markdown("""
<hr style="margin-top:40px;">
<div style="text-align:center; color:#374151; font-size:0.75rem;
            font-family:Space Mono,monospace; padding:12px 0;">
    CHURN PREDICTOR · XGBoost · scale_pos_weight=8.79 · seuil=0.40 · ROC-AUC=0.795
</div>
""", unsafe_allow_html=True)
