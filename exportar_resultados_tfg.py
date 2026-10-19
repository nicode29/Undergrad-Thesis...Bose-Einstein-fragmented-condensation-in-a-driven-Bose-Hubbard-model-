"""Exporta variables ya calculadas por TFG_refactorizacion.ipynb, sin recalcularlas."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np


def exportar_resultados(variables, carpeta):
    """Guarda NPZ sin pérdida y tablas TXT; nunca ejecuta el Hamiltoniano."""
    nombres = (
        "valores_kappa", "espectro", "energias_U", "H_U",
        "energias_fundamentales", "estados_fundamentales",
        "ks", "kappas", "R", "matrices_correlaciones",
    )
    datos = {
        nombre: np.asarray(variables[nombre])
        for nombre in nombres if nombre in variables
    }
    if not any(nombre in datos for nombre in (
        "espectro", "energias_fundamentales", "R", "matrices_correlaciones"
    )):
        raise ValueError("No hay resultados numéricos calculados en este kernel para exportar.")

    # Al terminar el barrido 8+8, el contexto restaura la base de seis sitios.
    base = variables.get("espacio_de_Fock")
    if base is not None:
        base = np.asarray(base)
        if base.ndim == 2 and base.shape[1] == 6 and np.all(base.sum(axis=1) == 6):
            datos["base_fock_6_bosones_6_sitios"] = base

    # Cada exportación tiene su carpeta para conservar las ejecuciones anteriores.
    destino = Path(carpeta) / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    destino.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(destino / "resultados_completos.npz", **datos)

    tablas = []

    def tabla(nombre, valores, cabecera, fmt="%.17g"):
        np.savetxt(destino / nombre, valores, delimiter="\t", header=cabecera,
                   fmt=fmt, encoding="utf-8")
        tablas.append(nombre)

    if "base_fock_6_bosones_6_sitios" in datos:
        base = datos["base_fock_6_bosones_6_sitios"]
        tabla("base_fock_6_bosones_6_sitios.txt",
              np.column_stack((np.arange(len(base)), base)),
              "indice_estado\t" + "\t".join(f"n_{j}" for j in range(6)), "%d")

    if "valores_kappa" in datos:
        kappa6 = datos["valores_kappa"]
        if "espectro" in datos:
            e = datos["espectro"]
            tabla("espectro_6_bosones_6_sitios.txt", np.column_stack((kappa6, e)),
                  "kappa\t" + "\t".join(f"E_{i}" for i in range(e.shape[1])))
        if "energias_fundamentales" in datos:
            tabla("energia_fundamental_6_bosones_6_sitios.txt",
                  np.column_stack((kappa6, datos["energias_fundamentales"])), "kappa\tE_0")
        if "estados_fundamentales" in datos:
            c = datos["estados_fundamentales"]
            tabla("estados_fundamentales_6_bosones_6_sitios.txt",
                  np.column_stack((np.repeat(kappa6, c.shape[1]),
                                   np.tile(np.arange(c.shape[1]), len(kappa6)),
                                   c.real.ravel(), c.imag.ravel())),
                  "kappa\tindice_estado\tamplitud_real\tamplitud_imag")

    if all(nombre in datos for nombre in ("kappas", "ks", "R")):
        kappa8, ks, r = datos["kappas"], datos["ks"], datos["R"]
        tabla("rho_8_bosones_8_sitios.txt",
              np.column_stack((np.repeat(kappa8, len(ks)), np.tile(ks, len(kappa8)),
                               np.tile(ks / np.pi, len(kappa8)), r.real.ravel(), r.imag.ravel())),
              "kappa\tk\tk_sobre_pi\trho_real\trho_imag")
    if all(nombre in datos for nombre in ("kappas", "matrices_correlaciones")):
        p = datos["matrices_correlaciones"]
        ik, j, l = np.indices(p.shape)
        tabla("correlaciones_8_bosones_8_sitios.txt",
              np.column_stack((datos["kappas"][ik.ravel()], j.ravel(), l.ravel(),
                               p.real.ravel(), p.imag.ravel())),
              "kappa\tj\tl\tP_real\tP_imag")

    metadatos = {
        "fecha": datetime.now().astimezone().isoformat(),
        "origen": "Variables existentes en el kernel; no se han recalculado resultados.",
        "caso_espectro_y_fundamentales": {"bosones": 6, "sitios": 6},
        "caso_rho_y_correlaciones": {"bosones": 8, "sitios": 8},
        "normalizacion_rho": "Sumatorio sin dividir por el número de sitios.",
        "estado_barrido_rho": variables.get("estado_ejecucion", "desconocido"),
        "variables_ausentes": [nombre for nombre in nombres if nombre not in datos],
        "arrays": {
            nombre: {"forma": list(a.shape), "dtype": str(a.dtype),
                     "elementos_no_finitos": int(np.count_nonzero(~np.isfinite(a)))}
            for nombre, a in datos.items()
        },
        "tablas": tablas,
        "no_disponible": (
            "El programa no conserva los estados ni las energías fundamentales de 8+8; "
            "no se exportan ni se reconstruyen."
        ),
    }
    (destino / "metadatos.json").write_text(
        json.dumps(metadatos, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (destino / "LEEME.txt").write_text(
        "resultados_completos.npz contiene los arrays originales, incluidos los complejos.\n"
        "Carga en Python: datos = np.load(r'RUTA/resultados_completos.npz', allow_pickle=False)\n"
        "Las tablas TXT están separadas por tabuladores, con índices desde cero.\n"
        "Las amplitudes y correlaciones complejas tienen columnas real e imaginaria.\n"
        "Consulta metadatos.json para saber qué variables existían y si el barrido terminó.\n",
        encoding="utf-8",
    )
    print(f"Datos exportados en: {destino}")
    return destino
