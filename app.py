"""Asistente de Mandarín — interfaz Streamlit."""
import os
from html import escape
from pathlib import Path

import streamlit as st
from openai import OpenAI

import core
import zona_profesor

st.set_page_config(page_title="Asistente de Mandarín", page_icon="🀄", layout="centered")

RAIZ = Path(__file__).parent

# ---------------------------------------------------------------------------
# Estilos
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap');
.block-container {max-width: 860px; padding-top: 3.6rem;}
.marca {display:flex; align-items:center; gap:.8rem; margin-bottom:.2rem;}
.sello {background:#C8102E; color:#fff; font-family:'Noto Sans SC',sans-serif; font-weight:700;
        font-size:1.35rem; line-height:1; padding:.5rem .55rem; border-radius:10px; letter-spacing:.05em;
        white-space:nowrap; flex-shrink:0;}
.marca h1 {font-size:1.7rem; margin:0; padding:0;}
.sub {color:#7A6A60; margin:0 0 1rem 0;}
.chip {display:inline-block; background:#F6E3E0; color:#8E0B20; border-radius:999px;
       padding:.15rem .7rem; font-size:.85rem; margin-right:.3rem; font-weight:600;}
.icono {font-size:1.9rem; line-height:1.2;}
.titulo-card {font-weight:700; font-size:1.05rem; margin:.15rem 0 .1rem;}
.desc-card {color:#7A6A60; font-size:.88rem; min-height:3.9em; margin-bottom:.3rem;}
.tarjeta {background:#fff; border:1px solid #EADFD3; border-left:5px solid #C8102E; border-radius:12px;
          padding:1rem 1.2rem; margin:.4rem 0 .8rem;}
.etq {text-transform:uppercase; letter-spacing:.08em; font-size:.72rem; color:#A08C7E; font-weight:700;}
.frase {font-size:1.35rem; font-weight:600; margin-top:.2rem; overflow-wrap:anywhere;}
.hanzi {font-family:'Noto Sans SC','PingFang SC','Microsoft YaHei',sans-serif; font-size:2rem;
        font-weight:500; letter-spacing:.06em; margin-top:.2rem; overflow-wrap:anywhere;}
.hanzi.texto {font-size:1.35rem; line-height:2.1rem;}
.pinyin {color:#8E0B20; font-size:1rem;}
.fichas {display:flex; flex-wrap:wrap; gap:.5rem; margin-top:.5rem;}
.ficha {font-family:'Noto Sans SC',sans-serif; font-size:1.4rem; background:#FBF3EA; border:1px solid #E6D6C4;
        border-radius:8px; padding:.25rem .7rem;}
.palabras {font-family:'Noto Sans SC',sans-serif; font-size:1.1rem; line-height:2rem; color:#3A2F2A;}
.glos {display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:.6rem;}
.glos div {background:#fff; border:1px solid #EADFD3; border-radius:10px; padding:.6rem .8rem;}
.glos b {font-family:'Noto Sans SC',sans-serif; font-size:1.5rem; font-weight:500; display:block;}
.glos i {color:#8E0B20; font-style:normal; font-size:.9rem; display:block;}
.glos span {color:#5E514A; font-size:.9rem;}
@media (max-width: 640px) {.desc-card {min-height:0;} .marca h1 {font-size:1.35rem;}}
.nota {font-size:2.6rem; font-weight:800; color:#C8102E; line-height:1;}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Datos del curso y cliente de DeepSeek
# ---------------------------------------------------------------------------
@st.cache_data
def cargar_curso():
    vocab = core.cargar_vocabulario((RAIZ / "vocabulario.txt").read_text(encoding="utf-8"))
    frases = core.cargar_frases((RAIZ / "training set.txt").read_text(encoding="utf-8"))
    return vocab, frases


def leer_secreto(nombre: str, defecto=None):
    """Lee un secreto de Streamlit Cloud (Settings → Secrets); si no hay, variable de entorno."""
    try:
        if nombre in st.secrets:
            return st.secrets[nombre]
        for clave in st.secrets:  # tolera otros nombres, p. ej. deepseek_api_key o [deepseek] api_key
            valor = st.secrets[clave]
            if nombre == "DEEPSEEK_API_KEY" and "deepseek" in clave.lower():
                if isinstance(valor, str):
                    return valor
                for sub in valor:
                    if "key" in sub.lower():
                        return valor[sub]
    except Exception:  # sin archivo de secretos (ejecución local)
        pass
    return os.getenv(nombre, defecto)


@st.cache_resource
def crear_tutor(clave: str, modelo: str) -> core.Tutor:
    return core.Tutor(OpenAI(api_key=clave, base_url="https://api.deepseek.com", timeout=60), modelo)


@st.cache_data(show_spinner=False, max_entries=500)
def audio_mp3(texto: str, voz: str, lento: bool) -> bytes:
    return core.sintetizar(texto, voz, lento)


@st.cache_data(show_spinner=False)
def glosario(palabras: tuple) -> list:
    return TUTOR.glosario(palabras)


VOCAB, FRASES = cargar_curso()
CLAVE = leer_secreto("DEEPSEEK_API_KEY")
if not CLAVE:
    st.error(
        "Falta la clave de DeepSeek. En Streamlit Cloud ve a **Settings → Secrets** y agrega:\n\n"
        '`DEEPSEEK_API_KEY = "sk-..."`'
    )
    st.stop()
TUTOR = crear_tutor(CLAVE, leer_secreto("DEEPSEEK_MODEL", "deepseek-chat"))

S = st.session_state
S.setdefault("pagina", "inicio")


def ir(pagina: str):
    S.pagina = pagina
    st.rerun()


def contexto() -> core.Contexto:
    return core.construir_contexto(VOCAB, FRASES, S.nivel, S.unidades)


def error_ia(err: Exception):
    st.error("No pude conectarme con el tutor en este momento. Intenta de nuevo en unos segundos.")
    st.caption(f"Detalle técnico: {type(err).__name__}")


def reproductor(texto: str, lento: bool = False, voz: str | None = None):
    try:
        st.audio(audio_mp3(texto, voz or core.VOCES["Voz femenina"], lento), format="audio/mp3")
    except Exception:
        st.warning("No se pudo generar el audio en este momento.")


def cabecera(titulo: str | None = None, desc: str | None = None):
    st.markdown(
        '<div class="marca"><span class="sello">汉语</span><h1>Asistente de Mandarín</h1></div>',
        unsafe_allow_html=True,
    )
    if S.pagina in ("inicio", "profesor"):
        return
    chips = f'<span class="chip">Chino {S.nivel}</span>' + "".join(
        f'<span class="chip">Vocabulario {u}</span>' for u in S.unidades
    )
    izq, c1, c2 = st.columns([5, 1.3, 1.9], vertical_alignment="center")
    izq.markdown(chips, unsafe_allow_html=True)
    if S.pagina != "menu" and c1.button("← Menú", use_container_width=True):
        ir("menu")
    if c2.button("Cambiar nivel", use_container_width=True):
        ir("inicio")
    if titulo:
        st.subheader(titulo)
    if desc:
        st.markdown(f'<p class="sub">{escape(desc)}</p>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Inicio: nivel y vocabulario
# ---------------------------------------------------------------------------
def pagina_inicio():
    cabecera()
    st.markdown(
        '<p class="sub">Practica con el vocabulario exacto de tu curso: traducciones, gramática, '
        "listening y exámenes de prueba.</p>",
        unsafe_allow_html=True,
    )
    niveles = sorted(VOCAB)
    st.markdown("#### 1 · ¿En qué nivel estás?")
    nivel = st.segmented_control(
        "Nivel", niveles, format_func=lambda n: f"Chino {n}", default=S.get("nivel", niveles[0]),
        label_visibility="collapsed", key="w_nivel",
    ) or niveles[0]

    unidades = sorted(VOCAB[nivel])
    st.markdown("#### 2 · ¿Qué vocabulario quieres repasar?")
    st.caption("Puedes elegir uno o varios. Los vocabularios anteriores se usan como apoyo en las frases.")
    previas = [u for u in S.get("unidades", ()) if u in unidades] if S.get("nivel") == nivel else []
    elegidas = st.pills(
        "Vocabulario", unidades, selection_mode="multi", format_func=lambda u: f"Vocabulario {u}",
        default=previas or unidades[-1:], label_visibility="collapsed", key=f"w_unidades_{nivel}",
    )
    if elegidas:
        with st.expander(f"Ver las palabras ({sum(len(VOCAB[nivel][u]) for u in elegidas)})"):
            for u in sorted(elegidas):
                st.markdown(
                    f'**Vocabulario {u}**<div class="palabras">{escape("  ·  ".join(VOCAB[nivel][u]))}</div>',
                    unsafe_allow_html=True,
                )
    st.write("")
    if st.button("Empezar a practicar →", type="primary", disabled=not elegidas, use_container_width=True):
        nuevo = (nivel, tuple(sorted(elegidas)))
        if nuevo != (S.get("nivel"), S.get("unidades")):
            for k in ("practica", "examen", "chat"):
                S.pop(k, None)
        S.nivel, S.unidades = nuevo
        ir("menu")
    if not elegidas:
        st.caption("Elige al menos un vocabulario para continuar.")
    st.divider()
    if st.button("🔒 Zona del profesor", key="ir_profesor"):
        ir("profesor")


def pagina_profesor():
    cabecera()
    if st.button("← Volver al inicio"):
        ir("inicio")
    zona_profesor.render(TUTOR, VOCAB, FRASES, RAIZ, leer_secreto, audio_mp3, cargar_curso.clear)


# ---------------------------------------------------------------------------
# Menú
# ---------------------------------------------------------------------------
ACTIVIDADES = [
    ("es_zh", "🇪🇸 → 🇨🇳", "Español → Chino", "Lee una frase en español y escríbela en chino."),
    ("zh_es", "🇨🇳 → 🇪🇸", "Chino → Español", "Lee una frase en chino y tradúcela al español."),
    ("escucha", "🎧", "Listening", "Escucha una frase en mandarín y escribe lo que oyes o qué significa."),
    ("chat", "💬", "Gramática con el tutor", "Pregunta dudas de gramática o cómo se dice algo, como en un chat."),
    ("examen", "📝", "Examen de prueba", "Elige la dificultad y mide cuánto sabes. Calificación al entregar."),
    ("vocab", "📖", "Mi vocabulario", "Repasa las palabras con pinyin, significado y pronunciación."),
]


def pagina_menu():
    cabecera()
    st.markdown("### ¿Qué quieres practicar hoy?")
    for fila in (ACTIVIDADES[:3], ACTIVIDADES[3:]):
        for col, (clave, icono, titulo, desc) in zip(st.columns(3), fila):
            with col.container(border=True):
                st.markdown(
                    f'<div class="icono">{icono}</div><div class="titulo-card">{titulo}</div>'
                    f'<div class="desc-card">{desc}</div>',
                    unsafe_allow_html=True,
                )
                if st.button("Empezar", key=f"ir_{clave}", use_container_width=True):
                    ir(clave)


# ---------------------------------------------------------------------------
# Práctica (traducción en ambos sentidos y listening)
# ---------------------------------------------------------------------------
MODOS = {
    "es_zh": ("Español → Chino", "Escribe la frase en chino (hanzi; si aún no puedes, en pinyin)."),
    "zh_es": ("Chino → Español", "Traduce la frase al español."),
    "escucha": ("Listening", "Escucha la frase las veces que necesites."),
}


def estado_practica(modo: str) -> dict:
    return S.setdefault("practica", {}).setdefault(
        modo, {"cola": [], "i": 0, "fase": "pregunta", "res": None, "vistas": [], "ok": 0.0, "total": 0, "n": 0}
    )


def tarjeta(etiqueta: str, texto: str, chino: bool = False, pinyin: str = ""):
    clase = "hanzi" if chino else "frase"
    extra = f'<div class="pinyin">{escape(pinyin)}</div>' if pinyin else ""
    st.markdown(
        f'<div class="tarjeta"><div class="etq">{escape(etiqueta)}</div>'
        f'<div class="{clase}">{escape(texto)}</div>{extra}</div>',
        unsafe_allow_html=True,
    )


def mostrar_resultado(res: dict):
    if res.get("sin_respuesta"):
        st.info("No pasa nada. Mira la respuesta, escúchala y sigue con la próxima.")
    elif res["puntaje"] >= 1:
        st.success(f"✅ ¡Correcto! {res['feedback']}")
    elif res["puntaje"] > 0:
        st.warning(f"🟡 Casi. {res['feedback']}")
    else:
        st.error(f"❌ {res['feedback']}")


def pagina_practica(modo: str):
    titulo, desc = MODOS[modo]
    cabecera(titulo, desc)
    e = estado_practica(modo)
    ctx = contexto()

    voz, lento, dictado = core.VOCES["Voz femenina"], False, True
    if modo == "escucha":
        a, b, c = st.columns(3)
        dictado = a.segmented_control("Ejercicio", ["Dictado", "Comprensión"], default="Dictado", key="w_tipo_esc") != "Comprensión"
        lento = b.segmented_control("Velocidad", ["Lenta", "Normal"], default="Lenta", key="w_vel") != "Normal"
        voz = core.VOCES[c.selectbox("Voz", list(core.VOCES), key="w_voz")]

    if e["i"] >= len(e["cola"]):
        with st.spinner("Preparando ejercicios nuevos…"):
            try:
                e["cola"], e["i"], e["fase"] = TUTOR.generar_frases(ctx, 6, e["vistas"]), 0, "pregunta"
            except Exception as err:
                error_ia(err)
                if st.button("Reintentar"):
                    st.rerun()
                return
    f = e["cola"][e["i"]]
    marcador = f"Ejercicio {e['total'] + (e['fase'] == 'pregunta')}"
    if e["total"]:
        marcador += f" · Aciertos: {e['ok']:g} de {e['total']}"
    st.caption(marcador)

    # --- enunciado (sin mostrar nunca la respuesta) ---
    if modo == "es_zh":
        tarjeta("Traduce al chino", f["es"])
        tarea, enunciado, ref = f"Traducir al chino (pinyin de la referencia: {f['pinyin']})", f["es"], f["zh"]
        pista = "Escribe en chino…"
    elif modo == "zh_es":
        tarjeta("Traduce al español", f["zh"], chino=True)
        tarea, enunciado, ref = "Traducir al español", f["zh"], f["es"]
        pista = "Escribe la traducción…"
    else:
        tarjeta("Escucha", "🎧 " + ("Escribe lo que oyes" if dictado else "¿Qué significa?"))
        reproductor(f["zh"], lento, voz)
        if dictado:
            tarea = f"Dictado: escuchó la frase y debía escribirla en hanzi o en pinyin (pinyin de la referencia: {f['pinyin']})"
            enunciado, ref, pista = "(audio)", f["zh"], "Escribe en hanzi o pinyin…"
        else:
            tarea, enunciado, ref = "Escuchó esta frase en chino y debía traducirla al español", f["zh"], f["es"]
            pista = "Escribe qué significa…"

    if e["fase"] == "pregunta":
        with st.form(f"f_{modo}_{e['n']}", border=False):
            resp = st.text_input("Tu respuesta", placeholder=pista, key=f"r_{modo}_{e['n']}")
            c1, c2 = st.columns([3, 1.2])
            comprobar = c1.form_submit_button("Comprobar", type="primary", use_container_width=True)
            no_se = c2.form_submit_button("No sé 🤷", use_container_width=True)
        if comprobar and not resp.strip():
            st.caption("Escribe tu respuesta, o pulsa **No sé** para verla.")
        elif comprobar or no_se:
            if no_se or core.es_no_se(resp):
                res = {"puntaje": 0.0, "feedback": "", "sin_respuesta": True}
            else:
                with st.spinner("Revisando tu respuesta…"):
                    try:
                        res = TUTOR.calificar(tarea, enunciado, ref, resp)
                    except Exception as err:
                        error_ia(err)
                        return
            res["respuesta"] = "" if res.get("sin_respuesta") else resp
            e.update(fase="resuelto", res=res, total=e["total"] + 1, ok=e["ok"] + res["puntaje"])
            e["vistas"].append(f["zh"])
            st.rerun()
    else:
        if e["res"]["respuesta"]:
            st.markdown(f"**Tu respuesta:** {escape(e['res']['respuesta'])}")
        mostrar_resultado(e["res"])
        tarjeta("Respuesta", f["zh"], chino=True, pinyin=f"{f['pinyin']}  —  {f['es']}")
        if modo != "escucha":
            reproductor(f["zh"])
        if st.button("Siguiente →", type="primary", use_container_width=True):
            e.update(i=e["i"] + 1, fase="pregunta", res=None, n=e["n"] + 1)
            st.rerun()


# ---------------------------------------------------------------------------
# Chat de gramática
# ---------------------------------------------------------------------------
SUGERENCIAS = {
    1: ["¿Cómo hago preguntas con 吗?", "¿Cuándo uso 不 y cuándo 没?", "¿Cuál es la diferencia entre 几 y 多少?"],
    2: ["¿Cómo se usa 的时候?", "¿Cómo comparo dos cosas con 比?", "¿Cuál es la diferencia entre 怎么 y 怎么样?"],
}


def pagina_chat():
    cabecera("Gramática con el tutor", "Pregunta lo que quieras sobre gramática o cómo decir algo con tu vocabulario.")
    hist = S.setdefault("chat", [])
    pregunta = st.chat_input("Escribe tu pregunta…")
    if not hist:
        st.caption("Ideas para empezar:")
        for col, sug in zip(st.columns(3), SUGERENCIAS.get(S.nivel, SUGERENCIAS[2])):
            if col.button(sug, use_container_width=True):
                pregunta = sug
    for m in hist:
        st.chat_message(m["role"], avatar="🧑‍🎓" if m["role"] == "user" else "🀄").markdown(m["content"])
    if pregunta:
        hist.append({"role": "user", "content": pregunta})
        st.chat_message("user", avatar="🧑‍🎓").markdown(pregunta)
        with st.chat_message("assistant", avatar="🀄"):
            try:
                respuesta = st.write_stream(TUTOR.chat(contexto(), hist))
                hist.append({"role": "assistant", "content": respuesta})
            except Exception as err:
                hist.pop()
                error_ia(err)
    if hist and st.button("🗑 Nueva conversación"):
        S.chat = []
        st.rerun()


# ---------------------------------------------------------------------------
# Examen de prueba
# ---------------------------------------------------------------------------
def items_examen(preguntas: list, respuestas: dict) -> list:
    """Aplana el examen en ítems calificables."""
    items = []
    for i, p in enumerate(preguntas):
        if p["tipo"] == "lectura":
            for j, s in enumerate(p["preguntas"]):
                items.append({
                    "id": f"{i}.{j}", "tarea": core.TIPOS["lectura"],
                    "enunciado": f"Texto: {p['texto']}\nPregunta: {s['enunciado']}",
                    "referencia": s["respuesta"], "respuesta": respuestas.get(f"{i}.{j}", ""),
                })
            continue
        if p["tipo"] == "ordenar":
            enunciado = " / ".join(p["palabras"])
        elif p["tipo"] == "redaccion":
            enunciado = f"Escribir unos {p['longitud']} caracteres usando: {'，'.join(p['palabras'])}"
        else:
            enunciado = p["enunciado"]
        items.append({
            "id": str(i), "tarea": core.TIPOS[p["tipo"]], "enunciado": enunciado,
            "referencia": p["respuesta"], "respuesta": respuestas.get(str(i), ""),
        })
    return items


def enunciado_examen(num: int, p: dict):
    st.markdown(f"**{num}. {core.TIPOS[p['tipo']]}**")
    if p["tipo"] in ("ordenar", "redaccion"):
        fichas = "".join(f'<span class="ficha">{escape(w)}</span>' for w in p["palabras"])
        st.markdown(f'<div class="fichas">{fichas}</div>', unsafe_allow_html=True)
        if p["tipo"] == "redaccion":
            st.caption(f"Escribe un texto de unos {p['longitud']} caracteres.")
    elif p["tipo"] == "lectura":
        st.markdown(f'<div class="tarjeta"><div class="hanzi texto">{escape(p["texto"])}</div></div>', unsafe_allow_html=True)
    elif p["tipo"] == "es_zh":
        st.markdown(f'<div class="frase">{escape(p["enunciado"])}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="hanzi">{escape(p["enunciado"])}</div>', unsafe_allow_html=True)


def pagina_examen():
    cabecera("Examen de prueba")
    ex = S.get("examen")

    # --- configuración ---
    if not ex:
        st.markdown('<p class="sub">Las respuestas y la calificación aparecen solo cuando entregas el examen.</p>', unsafe_allow_html=True)
        dificultad = st.segmented_control("Dificultad", list(core.DIFICULTADES), default="Fácil", key="w_dif") or "Fácil"
        st.caption(core.DIFICULTADES[dificultad]["ayuda"])
        n = st.select_slider("Número de preguntas", [5, 8, 10, 12], value=8, key="w_n")
        if st.button("Generar examen", type="primary", use_container_width=True):
            with st.spinner("Preparando tu examen…"):
                try:
                    preguntas = TUTOR.generar_examen(contexto(), dificultad, n)
                except Exception as err:
                    error_ia(err)
                    return
            S.examen = {"dificultad": dificultad, "preguntas": preguntas, "resultados": None, "respuestas": {},
                        "id": S.get("n_examen", 0)}
            S.n_examen = S.get("n_examen", 0) + 1
            st.rerun()
        return

    preguntas = ex["preguntas"]

    # --- resolviendo ---
    if ex["resultados"] is None:
        st.markdown(f'<span class="chip">Dificultad: {ex["dificultad"]}</span>', unsafe_allow_html=True)
        st.caption("Si no sabes una respuesta, déjala en blanco. Puedes responder en hanzi o en pinyin.")
        with st.form(f"examen_{ex['id']}"):
            resp = {}
            for i, p in enumerate(preguntas):
                enunciado_examen(i + 1, p)
                if p["tipo"] == "lectura":
                    for j, s in enumerate(p["preguntas"]):
                        resp[f"{i}.{j}"] = st.text_input(s["enunciado"], key=f"ex_{ex['id']}_{i}_{j}")
                elif p["tipo"] == "redaccion":
                    resp[str(i)] = st.text_area("Tu texto", key=f"ex_{ex['id']}_{i}", label_visibility="collapsed")
                else:
                    resp[str(i)] = st.text_input("Tu respuesta", key=f"ex_{ex['id']}_{i}", label_visibility="collapsed",
                                                 placeholder="Tu respuesta…")
                st.write("")
            entregar = st.form_submit_button("Entregar examen", type="primary", use_container_width=True)
        if entregar:
            with st.spinner("Calificando…"):
                try:
                    ex["resultados"] = TUTOR.calificar_examen(items_examen(preguntas, resp))
                except Exception as err:
                    error_ia(err)
                    return
            ex["respuestas"] = resp
            st.rerun()
        if st.button("Cancelar examen"):
            S.pop("examen")
            st.rerun()
        return

    # --- resultados ---
    res, items = ex["resultados"], items_examen(preguntas, ex["respuestas"])
    puntos, total = sum(res[it["id"]]["puntaje"] for it in items), len(items)
    pct = round(100 * puntos / total)
    mensaje = "¡Excelente! 太好了" if pct >= 85 else "¡Vas bien! Repasa los errores." if pct >= 60 else "Sigue practicando: revisa cada explicación."
    st.markdown(
        f'<div class="tarjeta"><div class="etq">Resultado · {ex["dificultad"]}</div>'
        f'<div class="nota">{pct}%</div><div>{puntos:g} de {total} puntos — {mensaje}</div></div>',
        unsafe_allow_html=True,
    )
    for i, p in enumerate(preguntas):
        with st.container(border=True):
            enunciado_examen(i + 1, p)
            subs = [(f"{i}.{j}", s["enunciado"], s["respuesta"]) for j, s in enumerate(p["preguntas"])] if p["tipo"] == "lectura" \
                else [(str(i), None, p["respuesta"])]
            for clave, sub, correcta in subs:
                r = res[clave]
                if sub:
                    st.markdown(f"**{escape(sub)}**")
                icono = "✅" if r["puntaje"] >= 1 else "🟡" if r["puntaje"] > 0 else "❌"
                tuya = ex["respuestas"].get(clave, "").strip() or "(en blanco)"
                st.markdown(f"{icono} **Tu respuesta:** {escape(tuya)}")
                etiqueta = "Texto modelo" if p["tipo"] == "redaccion" else "Respuesta correcta"
                if r["puntaje"] < 1 or p["tipo"] == "redaccion":
                    st.markdown(f"**{etiqueta}:** {escape(correcta)}")
                if r["feedback"] and not r.get("sin_respuesta"):
                    st.caption(r["feedback"])
    if st.button("Hacer otro examen", type="primary", use_container_width=True):
        S.pop("examen")
        st.rerun()


# ---------------------------------------------------------------------------
# Mi vocabulario
# ---------------------------------------------------------------------------
def pagina_vocab():
    cabecera("Mi vocabulario", "Las palabras que elegiste, con pinyin y significado.")
    ctx = contexto()
    try:
        with st.spinner("Preparando tu glosario…"):
            entradas = glosario(ctx.foco)
    except Exception:
        entradas = [{"zh": p, "pinyin": "", "es": ""} for p in ctx.foco]
    celdas = "".join(
        f'<div><b>{escape(str(g["zh"]))}</b><i>{escape(str(g.get("pinyin", "")))}</i>'
        f'<span>{escape(str(g.get("es", "")))}</span></div>'
        for g in entradas
    )
    st.markdown(f'<div class="glos">{celdas}</div>', unsafe_allow_html=True)
    st.write("")
    palabra = st.selectbox("🔊 Escuchar una palabra", ctx.foco, index=None, placeholder="Elige una palabra…")
    if palabra:
        reproductor(palabra, lento=True)


# ---------------------------------------------------------------------------
# Navegación
# ---------------------------------------------------------------------------
if S.pagina not in ("inicio", "profesor") and not S.get("unidades"):
    S.pagina = "inicio"

if S.pagina == "inicio":
    pagina_inicio()
elif S.pagina == "menu":
    pagina_menu()
elif S.pagina in MODOS:
    pagina_practica(S.pagina)
elif S.pagina == "chat":
    pagina_chat()
elif S.pagina == "examen":
    pagina_examen()
elif S.pagina == "vocab":
    pagina_vocab()
elif S.pagina == "profesor":
    pagina_profesor()
