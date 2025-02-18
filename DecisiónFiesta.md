# Modelo de Decisión para Salir de Fiesta

## Decisiones (D):
- d₀: No_Salir
- d₁: Salir_Sin_Tomar
- d₂: Salir_Y_Tomar_Moderado
- d₃: Salir_Sin_Preocupaciones

## Estados de la naturaleza (Ω):
- ω₀: Gran_Fiesta_Sin_Excesos
- ω₁: Gran_Fiesta_Muy_Ebrio
- ω₂: Mala_Fiesta_Muy_Ebrio
- ω₃: Fiesta_Normal_Sin_Excesos
- ω₄: Gran_Fiesta_Incumplimiento
- ω₅: Mala_Fiesta_Incumplimiento
- ω₆: Accidente_O_Problema_Grave
- ω₇: Fiesta_Aburrida
- ω₈: Problema_Salud
- ω₉: Problema_Economico

## Información proxy (Z):
Vector de observaciones previas a la decisión:

1. Estado actual [vector]:
   - Nivel de alcohol [0-1]
   - Nivel de cansancio [0-1]
   - Estado de salud [0-1]
   - Deuda de sueño [0-1]
   - Ya cené [0,1]

2. Características del lugar [vector]:
   - Música agradable [0-1]
   - Lugar agradable [0-1]
   - Distancia al lugar [0-1]
   - Lluvia [0,1]
   - Temperatura inadecuada [0,1]

3. Factores sociales [vector]:
   - Cantidad de amigos [0-1]
   - Presión social [0-1]
   - Ocasion especial [0,1]
   - FOMO (miedo a perderse algo) [0-1]

4. Logística [vector]:
   - Trabajo mañana [0,1]
   - Transporte seguro [0,1]
   - Hora actual [time]
   - Hora de regreso [time]
   - Gasto esperado [0-1]
   - Presupuesto [0-1]
   - Costo de transporte [0-1]
   - Cover [0-1]

## Función de utilidad U(ω,d):

Estado (ω)               | No_Salir (d₀) | Salir_Sin_Tomar (d₁) | Salir_Y_Tomar_Moderado (d₂) | Salir_Sin_Preocupaciones (d₃)
--------------------------|---------------|-----------------------|-----------------------------|------------------------------
ω₀ (Gran_Fiesta_Sin_Excesos) | 0            | 8                     | 15                          | 40
ω₁ (Gran_Fiesta_Muy_Ebrio)   | 0            | -5                    | -8                          | 10
ω₂ (Mala_Fiesta_Muy_Ebrio)   | 0            | -10                   | -15                         | -5
ω₃ (Fiesta_Normal_Sin_Excesos)| 0           | 5                     | 50                          | 8
ω₄ (Gran_Fiesta_Incumplimiento)| 0          | 55                    | 5                           | -8
ω₅ (Mala_Fiesta_Incumplimiento)| 0          | -8                    | -12                         | -15
ω₆ (Accidente_O_Problema_Grave)| 0          | -20                   | -25                         | -30
ω₇ (Fiesta_Aburrida)         | 0            | -10                   | -15                         | -20
ω₈ (Problema_Salud)          | 0            | -25                   | -30                         | -35
ω₉ (Problema_Economico)      | 0            | -8                    | -12                         | -15

## Reglas de decisión basadas en proxies:

### Restricciones absolutas (si cualquiera es verdadera → d₀):
- Nivel de alcohol > 0.5 Y transporte seguro = 0
- Hora de regreso >= 2:00 AM Y trabajo mañana = 1
- Riesgo de salud > 0.8
- Riesgo económico > 0.9

### Indicadores de alto riesgo (aumentan prob. de ω₁, ω₂, ω₆, ω₈):
- Nivel de alcohol > 0.5
- Deuda de sueño > 0.7
- Lluvia = 1 Y distancia al lugar > 0.7
- Estado de salud < 0.5 Y medicamentos = 1

### Indicadores positivos (aumentan prob. de ω₀, ω₃, ω₄):
- Cantidad de amigos > 0.7
- Música agradable > 0.7 Y lugar agradable > 0.7
- Presión social > 0.6 Y ocasion especial = 1
- Transporte seguro = 1 Y hora temprana

## Heurística simplificada:
1. Verificar restricciones absolutas primero.
2. Si hay 2+ indicadores de alto riesgo → No_Salir (d₀).
3. Evaluar balance entre indicadores positivos y negativos.
4. Considerar estado actual:
   - Si nivel de alcohol > 0.5 → Peso extra a riesgos.
   - Si deuda de sueño > 0.7 → Peso extra a cansancio.
   - Si riesgo económico > 0.7 → Peso extra a limitaciones financieras.
5. Calcular valores esperados y elegir la decisión con mayor utilidad.


```python
import numpy as np
from datetime import datetime, time
from typing import Dict, List, Union

class FiestaDecisionAvanzada:
    def __init__(self):
        # Estados posibles
        self.estados = [
            "Gran_Fiesta_Sin_Excesos",      # La mejor noche, todo controlado
            "Gran_Fiesta_Muy_Ebrio",        # Buena noche pero exceso de alcohol
            "Mala_Fiesta_Muy_Ebrio",        # Noche mala y exceso de alcohol
            "Fiesta_Normal_Sin_Excesos",     # Noche normal, todo controlado
            "Gran_Fiesta_Incumplimiento",    # Buena noche pero problemas mañana
            "Mala_Fiesta_Incumplimiento",    # Mala noche y problemas mañana
            "Accidente_O_Problema_Grave",    # Situaciones peligrosas
            "Fiesta_Aburrida",              # Noche sin gracia
            "Problema_Salud",               # Problemas de salud
            "Problema_Economico"            # Problemas financieros
        ]
        
        self.decisiones = [
            "No_Salir",
            "Salir_Sin_Tomar",
            "Salir_Y_Tomar_Moderado",
            "Salir_Sin_Preocupaciones"
        ]
        
        # Matriz de utilidad expandida
        self.U = np.array([
            [0,  8,  15,  40],    # Gran_Fiesta_Sin_Excesos
            [0,  -5,  -8,  10],    # Gran_Fiesta_Muy_Ebrio
            [0, -10, -15,  -5],    # Mala_Fiesta_Muy_Ebrio
            [0,   5,  50,   8],    # Fiesta_Normal_Sin_Excesos
            [0,   55,   5,  -8],    # Gran_Fiesta_Incumplimiento
            [0,  -8, -12, -15],    # Mala_Fiesta_Incumplimiento
            [0, -20, -25, -30],    # Accidente_O_Problema_Grave
            [0,  -10,  -15, -20],    # Fiesta_Aburrida
            [0, -25, -30, -35],    # Problema_Salud
            [0,  -8, -12, -15]     # Problema_Economico
        ])
        
        # Probabilidades base ajustadas
        self.p_base = np.array([0.15, 0.10, 0.10, 0.20, 0.10, 0.10, 0.05, 0.10, 0.05, 0.05])
    
    def calcular_factor_social(self, proxies: Dict) -> Dict[str, float]:
        """Calcula el impacto de factores sociales"""
        presion_social = proxies.get('presion_social', 0.0)
        ocasion_especial = proxies.get('ocasion_especial', False)
        fomo = proxies.get('fomo', 0.0)
        
        factor_social = presion_social * 0.4 + (0.3 if ocasion_especial else 0.0) + fomo * 0.3
        return {
            'factor_fiesta': factor_social,
            'factor_exceso': factor_social * 0.6
        }
    
    def calcular_factor_clima(self, proxies: Dict) -> Dict[str, float]:
        """Calcula el impacto del clima"""
        lluvia = proxies.get('lluvia', 0.0)
        temperatura_inadecuada = proxies.get('temperatura_inadecuada', 0.0)
        distancia_venue = proxies.get('distancia_venue', 0.0)
        
        factor_riesgo = lluvia * 0.4 + temperatura_inadecuada * 0.3 + distancia_venue * 0.3
        return {
            'factor_riesgo': factor_riesgo,
            'factor_aburrimiento': factor_riesgo * 0.4
        }
    
    def calcular_factor_salud(self, proxies: Dict) -> Dict[str, float]:
        """Calcula el impacto de factores de salud"""
        estado_salud = proxies.get('estado_salud', 1.0)
        medicamentos = proxies.get('medicamentos', False)
        calidad_comida = proxies.get('calidad_comida', 0.5)
        deuda_sueño = proxies.get('deuda_sueño', 0.0)
        
        riesgo_salud = (
            (1 - estado_salud) * 0.4 +
            (0.3 if medicamentos else 0.0) +
            (1 - calidad_comida) * 0.1 +
            deuda_sueño * 0.2
        )
        return {
            'riesgo_salud': riesgo_salud,
            'factor_resistencia': 1 - riesgo_salud
        }
    
    def calcular_factor_economico(self, proxies: Dict) -> Dict[str, float]:
        """Calcula el impacto económico"""
        gasto_esperado = proxies.get('gasto_esperado', 0.0)
        presupuesto = proxies.get('presupuesto', 1.0)
        costo_transporte = proxies.get('costo_transporte', 0.0)
        cover = proxies.get('cover', 0.0)
        
        gasto_total = (gasto_esperado + costo_transporte + cover) / presupuesto
        return {
            'riesgo_economico': max(0, min(1, gasto_total)),
            'factor_limitacion': gasto_total > 0.7
        }
    
    def ajustar_probabilidades(self, proxies: Dict) -> np.ndarray:
        """Ajusta probabilidades según el contexto completo"""
        prob = self.p_base.copy()
        
        # Factores básicos originales
        if proxies['nivel_alcohol'] > 0.5:
            prob[1] += 0.2  # Más probable acabar muy ebrio
            prob[2] += 0.15  # Más probable mala fiesta y muy ebrio
            prob[6] += 0.1  # Más riesgo de accidentes
            prob[4] += 0.2
            prob[5] += 0.2
        elif 0.3 <= proxies['nivel_alcohol'] <= 0.5:
            prob[0] += 0.15  # Más probable gran fiesta sin excesos
            prob[3] += 0.1   # Más probable fiesta normal
        
        # Nuevos factores sociales
        factores_sociales = self.calcular_factor_social(proxies)
        if factores_sociales['factor_fiesta'] > 0.6:
            prob[0] += 0.15  # Más probable gran fiesta
            prob[1] += factores_sociales['factor_exceso'] * 0.2  # Más probable excesos
        
        # Factores climáticos
        factores_clima = self.calcular_factor_clima(proxies)
        if factores_clima['factor_riesgo'] > 0.5:
            prob[6] += 0.3  # Más riesgo de accidentes
            prob[7] += factores_clima['factor_aburrimiento'] * 0.2  # Más probable aburrimiento
        
        # Factores de salud
        factores_salud = self.calcular_factor_salud(proxies)
        if factores_salud['riesgo_salud'] > 0.6:
            prob[8] += 0.2  # Más probable problema de salud
            prob[2] += 0.15  # Más probable mala fiesta con excesos
        
        # Factores económicos
        factores_economicos = self.calcular_factor_economico(proxies)
        if factores_economicos['riesgo_economico'] > 0.7:
            prob[9] += 0.4  # Más probable problema económico
        
        # Ajustes adicionales originales
        if proxies['cantidad_amigos'] > 0.7:
            prob[0] += 0.15
            prob[1] += 0.1
            prob[4] += 0.1
        if proxies['trabajo_mañana']:
            prob[4] += 0.2
            prob[5] += 0.2
        if proxies['musica_agradable'] > 0.7 and proxies['lugar_agradable'] > 0.7:
            prob[0] += 0.2
            prob[3] += 0.15
            prob[7] -= 0.1
        
        # Normalizar probabilidades
        return prob / np.sum(prob)
    
    def verificar_restricciones(self, proxies: Dict) -> Dict[str, List[str]]:
        """Verifica restricciones y genera advertencias"""
        restricciones = []
        advertencias = []
        
        # Restricciones originales
        if proxies['nivel_alcohol'] > 0.5 and not proxies['transporte_seguro']:
            restricciones.append("Nivel de alcohol alto sin transporte seguro")
        
        if proxies['trabajo_mañana'] and proxies['hora_regreso'].hour >= 2:
            restricciones.append("Hora de regreso muy tarde con trabajo mañana")
        
        # Nuevas restricciones
        factores_salud = self.calcular_factor_salud(proxies)
        if factores_salud['riesgo_salud'] > 0.8:
            restricciones.append("Riesgo de salud demasiado alto")
        
        factores_economicos = self.calcular_factor_economico(proxies)
        if factores_economicos['riesgo_economico'] > 0.9:
            restricciones.append("Riesgo económico demasiado alto")
        
        # Nuevas advertencias
        if proxies.get('medicamentos', False):
            advertencias.append("Medicamentos pueden interactuar con alcohol")
        
        if proxies.get('deuda_sueño', 0.0) > 0.7:
            advertencias.append("Alto nivel de deuda de sueño")
        
        if factores_economicos['factor_limitacion']:
            advertencias.append("Gastos cercanos al límite del presupuesto")
        
        return {
            'restricciones': restricciones,
            'advertencias': advertencias
        }
    
    def sugerir_decision(self, proxies: Dict) -> Dict:
        """Analiza la situación y sugiere la mejor decisión"""
        restricciones_info = self.verificar_restricciones(proxies)
         if len(resultado['restricciones']) > 2:
            return {
            'decision': 'No_Salir',
            'restricciones': restricciones_info['restricciones'],
            'advertencias': restricciones_info['advertencias'],
            'explicacion': "No es posible salir",
            'valores_esperados': np.zeros(len(self.decisiones)),
            'probabilidades': self.p_base
            }
        # Calculate valores_esperados regardless of restrictions
        prob_ajustadas = self.ajustar_probabilidades(proxies)
        valores_esperados = np.zeros(len(self.decisiones))
        
        for i in range(len(self.decisiones)):
            valores_esperados[i] = np.sum(prob_ajustadas * self.U[:, i])
        
        
        
        mejor_decision_idx = np.argmax(valores_esperados)
        explicacion = self.generar_explicacion(proxies, valores_esperados)
        
        return {
            'decision': self.decisiones[mejor_decision_idx],
            'restricciones': restricciones_info['restricciones'],
            'advertencias': restricciones_info['advertencias'],
            'explicacion': explicacion,
            'valores_esperados': valores_esperados,
            'probabilidades': prob_ajustadas
        }
    
    def generar_explicacion(self, proxies: Dict, valores_esperados: np.ndarray) -> str:
        """Genera una explicación detallada de la decisión"""
        explicacion = []
        
        # Factores básicos
        if proxies['cantidad_amigos'] > 0.7:
            explicacion.append("Hay muchos amigos")
        
        if proxies['musica_agradable'] > 0.7 and proxies['lugar_agradable'] > 0.7:
            explicacion.append("Ambiente muy favorable")
        
        # Nuevos factores
        if proxies.get('presion_social', 0.0) > 0.7:
            explicacion.append("Alta presión social")
        
        if proxies.get('ocasion_especial', False):
            explicacion.append("Es una ocasión especial")
        
        if proxies.get('deuda_sueño', 0.0) > 0.6:
            explicacion.append("Considerable deuda de sueño")
        
        factores_economicos = self.calcular_factor_economico(proxies)
        if factores_economicos['riesgo_economico'] > 0.6:
            explicacion.append("Alto riesgo económico")
        
        return " | ".join(explicacion)

# Ejemplo de uso con nuevos factores
fiesta = FiestaDecisionAvanzada()

proxies_ejemplo = {
    # Factores originales
    'musica_agradable': 0,
    'lugar_agradable': 0,
    'nivel_alcohol': 1,
    'cantidad_amigos': 0,
    'nivel_cansancio': 1,
    'trabajo_mañana': True,
    'transporte_seguro': False,
    'ya_cenaste': False,
    'hora': datetime.strptime("23:30", "%H:%M").time(),
    'hora_regreso': datetime.strptime("02:30", "%H:%M").time(),
    
    # Nuevos factores sociales
    'presion_social': 0.3,
    'ocasion_especial': False,
    'fomo': 0.7,
    
    # Factores climáticos
    'lluvia': 1,
    'temperatura_inadecuada': 0,
    'distancia_venue': 1,
    
    # Factores de salud
    'estado_salud': 0.1,
    'medicamentos': True,
    'calidad_comida': 0.1,
    'deuda_sueño': 1,
    
    # Factores económicos
    'gasto_esperado': .5,
    'presupuesto': 1,
    'costo_transporte': 0.2,
    'cover': 0.1
}

resultado = fiesta.sugerir_decision(proxies_ejemplo)

print("\n=== ANÁLISIS AVANZADO DE DECISIÓN PARA FIESTA ===")
print("\nCONDICIONES BÁSICAS:")
print(f"Hora actual: {proxies_ejemplo['hora'].strftime('%H:%M')}")
print(f"Hora prevista regreso: {proxies_ejemplo['hora_regreso'].strftime('%H:%M')}")
print(f"Nivel de alcohol: {proxies_ejemplo['nivel_alcohol']*100:.0f}%")
print(f"Cantidad de amigos: {proxies_ejemplo['cantidad_amigos']*100:.0f}%")
print(f"¿Ya cenaste?: {'Sí' if proxies_ejemplo['ya_cenaste'] else 'No'}")
print(f"Trabajo mañana: {'Sí' if proxies_ejemplo['trabajo_mañana'] else 'No'}")

print("\nFACTORES SOCIALES:")
print(f"Presión social: {proxies_ejemplo['presion_social']*100:.0f}%")
print(f"Ocasión especial: {'Sí' if proxies_ejemplo['ocasion_especial'] else 'No'}")
print(f"FOMO: {proxies_ejemplo['fomo']*100:.0f}%")

print("\nFACTORES AMBIENTALES:")
print(f"Música agradable: {proxies_ejemplo['musica_agradable']*100:.0f}%")
print(f"Lugar agradable: {proxies_ejemplo['lugar_agradable']*100:.0f}%")
print(f"Lluvia: {proxies_ejemplo['lluvia']*100:.0f}%")
print(f"Temperatura inadecuada: {proxies_ejemplo['temperatura_inadecuada']*100:.0f}%")
print(f"Distancia al lugar: {proxies_ejemplo['distancia_venue']*100:.0f}%")

print("\nFACTORES DE SALUD:")
print(f"Estado de salud: {proxies_ejemplo['estado_salud']*100:.0f}%")
print(f"Medicamentos: {'Sí' if proxies_ejemplo['medicamentos'] else 'No'}")
print(f"Calidad de comida: {proxies_ejemplo['calidad_comida']*100:.0f}%")
print(f"Deuda de sueño: {proxies_ejemplo['deuda_sueño']*100:.0f}%")

print("\nFACTORES ECONÓMICOS:")
print(f"Gasto esperado: {proxies_ejemplo['gasto_esperado']*100:.0f}%")
print(f"Presupuesto disponible: {proxies_ejemplo['presupuesto']*100:.0f}%")
print(f"Costo transporte: {proxies_ejemplo['costo_transporte']*100:.0f}%")
print(f"Cover: {proxies_ejemplo['cover']*100:.0f}%")

print("\nDECISIÓN SUGERIDA:", resultado['decision'].replace('_', ' '))
print("\nEXPLICACIÓN:", resultado['explicacion'])

if resultado['advertencias']:
    print("\nADVERTENCIAS:")
    for adv in resultado['advertencias']:
        print(f"⚠️ {adv}")

if resultado['restricciones']:
    print("\nRESTRICCIONES CRÍTICAS:")
    for res in resultado['restricciones']:
        print(f"🚫 {res}")

print("\nPROBABILIDADES DE CADA ESCENARIO:")
for estado, prob in zip(fiesta.estados, resultado.get('probabilidades', fiesta.p_base)):
    estado_fmt = estado.replace('_', ' ').ljust(30)
    prob_pct = f"{(prob*100):5.1f}%"
    barra = "█" * int(prob * 40)
    print(f"{estado_fmt} {prob_pct} {barra}")

print("\nVALORES ESPERADOS POR DECISIÓN:")
valores = resultado.get('valores_esperados', np.zeros(len(fiesta.decisiones)))
for decision, valor in zip(fiesta.decisiones, valores):
    decision_fmt = decision.replace('_', ' ').ljust(30)
    valor_fmt = f"{valor:6.2f}"
    print(f"{decision_fmt} {valor_fmt}")

# Análisis de factores críticos
print("\nFACTORES CRÍTICOS:")
factores_salud = fiesta.calcular_factor_salud(proxies_ejemplo)
factores_economicos = fiesta.calcular_factor_economico(proxies_ejemplo)
factores_sociales = fiesta.calcular_factor_social(proxies_ejemplo)
factores_clima = fiesta.calcular_factor_clima(proxies_ejemplo)

print(f"Riesgo de salud: {factores_salud['riesgo_salud']*100:.1f}%")
print(f"Riesgo económico: {factores_economicos['riesgo_economico']*100:.1f}%")
print(f"Factor social: {factores_sociales['factor_fiesta']*100:.1f}%")
print(f"Riesgo climático: {factores_clima['factor_riesgo']*100:.1f}%")

# Resumen final
print("\nRESUMEN FINAL:")
print("=" * 50)
print(f"Decisión: {resultado['decision'].replace('_', ' ')}")
print(f"Confianza: {max(resultado['valores_esperados'])/20*100:.1f}%")
print("=" * 50)
```

    
    === ANÁLISIS AVANZADO DE DECISIÓN PARA FIESTA ===
    
    CONDICIONES BÁSICAS:
    Hora actual: 23:30
    Hora prevista regreso: 02:30
    Nivel de alcohol: 100%
    Cantidad de amigos: 0%
    ¿Ya cenaste?: No
    Trabajo mañana: Sí
    
    FACTORES SOCIALES:
    Presión social: 30%
    Ocasión especial: No
    FOMO: 70%
    
    FACTORES AMBIENTALES:
    Música agradable: 0%
    Lugar agradable: 0%
    Lluvia: 100%
    Temperatura inadecuada: 0%
    Distancia al lugar: 100%
    
    FACTORES DE SALUD:
    Estado de salud: 10%
    Medicamentos: Sí
    Calidad de comida: 10%
    Deuda de sueño: 100%
    
    FACTORES ECONÓMICOS:
    Gasto esperado: 50%
    Presupuesto disponible: 100%
    Costo transporte: 20%
    Cover: 10%
    
    DECISIÓN SUGERIDA: No Salir
    
    EXPLICACIÓN: Considerable deuda de sueño | Alto riesgo económico
    
    ADVERTENCIAS:
    ⚠️ Medicamentos pueden interactuar con alcohol
    ⚠️ Alto nivel de deuda de sueño
    ⚠️ Gastos cercanos al límite del presupuesto
    
    RESTRICCIONES CRÍTICAS:
    🚫 Nivel de alcohol alto sin transporte seguro
    🚫 Hora de regreso muy tarde con trabajo mañana
    🚫 Riesgo de salud demasiado alto
    
    PROBABILIDADES DE CADA ESCENARIO:
    Gran Fiesta Sin Excesos          4.5% █
    Gran Fiesta Muy Ebrio            8.9% ███
    Mala Fiesta Muy Ebrio           11.9% ████
    Fiesta Normal Sin Excesos        6.0% ██
    Gran Fiesta Incumplimiento      14.9% █████
    Mala Fiesta Incumplimiento      14.9% █████
    Accidente O Problema Grave      13.4% █████
    Fiesta Aburrida                  4.6% █
    Problema Salud                   7.4% ██
    Problema Economico              13.4% █████
    
    VALORES ESPERADOS POR DECISIÓN:
    No Salir                         0.00
    Salir Sin Tomar                 -0.06
    Salir Y Tomar Moderado          -7.79
    Salir Sin Preocupaciones       -10.44
    
    FACTORES CRÍTICOS:
    Riesgo de salud: 95.0%
    Riesgo económico: 80.0%
    Factor social: 33.0%
    Riesgo climático: 70.0%
    
    RESUMEN FINAL:
    ==================================================
    Decisión: No Salir
    Confianza: 0.0%
    ==================================================


### Salir Sin Preocupaciones


```python
fiesta = FiestaDecisionAvanzada()

proxies_ejemplo = {
    # Factores originales
    'musica_agradable': 1,
    'lugar_agradable': 1,
    'nivel_alcohol': 0.4,
    'cantidad_amigos': 1,
    'nivel_cansancio': 0.1,
    'trabajo_mañana': False,
    'transporte_seguro': True,
    'ya_cenaste': True,
    'hora': datetime.strptime("23:30", "%H:%M").time(),
    'hora_regreso': datetime.strptime("02:30", "%H:%M").time(),
    
    # Nuevos factores sociales
    'presion_social': 1,
    'ocasion_especial': False,
    'fomo': 1,
    
    # Factores climáticos
    'lluvia': 0.1,
    'temperatura_inadecuada': 0.9,
    'distancia_venue': 0.1,
    
    # Factores de salud
    'estado_salud': 0.9,
    'medicamentos': False,
    'calidad_comida': 1,
    'deuda_sueño': 0.1,
    
    # Factores económicos
    'gasto_esperado': 0.1,
    'presupuesto': 1,
    'costo_transporte': 0.1,
    'cover': 0.1
}

resultado = fiesta.sugerir_decision(proxies_ejemplo)

print("\n=== ANÁLISIS AVANZADO DE DECISIÓN PARA FIESTA ===")
print("\nCONDICIONES BÁSICAS:")
print(f"Hora actual: {proxies_ejemplo['hora'].strftime('%H:%M')}")
print(f"Hora prevista regreso: {proxies_ejemplo['hora_regreso'].strftime('%H:%M')}")
print(f"Nivel de alcohol: {proxies_ejemplo['nivel_alcohol']*100:.0f}%")
print(f"Cantidad de amigos: {proxies_ejemplo['cantidad_amigos']*100:.0f}%")
print(f"¿Ya cenaste?: {'Sí' if proxies_ejemplo['ya_cenaste'] else 'No'}")
print(f"Trabajo mañana: {'Sí' if proxies_ejemplo['trabajo_mañana'] else 'No'}")

print("\nFACTORES SOCIALES:")
print(f"Presión social: {proxies_ejemplo['presion_social']*100:.0f}%")
print(f"Ocasión especial: {'Sí' if proxies_ejemplo['ocasion_especial'] else 'No'}")
print(f"FOMO: {proxies_ejemplo['fomo']*100:.0f}%")

print("\nFACTORES AMBIENTALES:")
print(f"Música agradable: {proxies_ejemplo['musica_agradable']*100:.0f}%")
print(f"Lugar agradable: {proxies_ejemplo['lugar_agradable']*100:.0f}%")
print(f"Lluvia: {proxies_ejemplo['lluvia']*100:.0f}%")
print(f"Temperatura inadecuada: {proxies_ejemplo['temperatura_inadecuada']*100:.0f}%")
print(f"Distancia al lugar: {proxies_ejemplo['distancia_venue']*100:.0f}%")

print("\nFACTORES DE SALUD:")
print(f"Estado de salud: {proxies_ejemplo['estado_salud']*100:.0f}%")
print(f"Medicamentos: {'Sí' if proxies_ejemplo['medicamentos'] else 'No'}")
print(f"Calidad de comida: {proxies_ejemplo['calidad_comida']*100:.0f}%")
print(f"Deuda de sueño: {proxies_ejemplo['deuda_sueño']*100:.0f}%")

print("\nFACTORES ECONÓMICOS:")
print(f"Gasto esperado: {proxies_ejemplo['gasto_esperado']*100:.0f}%")
print(f"Presupuesto disponible: {proxies_ejemplo['presupuesto']*100:.0f}%")
print(f"Costo transporte: {proxies_ejemplo['costo_transporte']*100:.0f}%")
print(f"Cover: {proxies_ejemplo['cover']*100:.0f}%")

print("\nDECISIÓN SUGERIDA:", resultado['decision'].replace('_', ' '))
print("\nEXPLICACIÓN:", resultado['explicacion'])

if resultado['advertencias']:
    print("\nADVERTENCIAS:")
    for adv in resultado['advertencias']:
        print(f"⚠️ {adv}")

if resultado['restricciones']:
    print("\nRESTRICCIONES CRÍTICAS:")
    for res in resultado['restricciones']:
        print(f"🚫 {res}")

print("\nPROBABILIDADES DE CADA ESCENARIO:")
for estado, prob in zip(fiesta.estados, resultado.get('probabilidades', fiesta.p_base)):
    estado_fmt = estado.replace('_', ' ').ljust(30)
    prob_pct = f"{(prob*100):5.1f}%"
    barra = "█" * int(prob * 40)
    print(f"{estado_fmt} {prob_pct} {barra}")

print("\nVALORES ESPERADOS POR DECISIÓN:")
valores = resultado.get('valores_esperados', np.zeros(len(fiesta.decisiones)))
for decision, valor in zip(fiesta.decisiones, valores):
    decision_fmt = decision.replace('_', ' ').ljust(30)
    valor_fmt = f"{valor:6.2f}"
    print(f"{decision_fmt} {valor_fmt}")

# Análisis de factores críticos
print("\nFACTORES CRÍTICOS:")
factores_salud = fiesta.calcular_factor_salud(proxies_ejemplo)
factores_economicos = fiesta.calcular_factor_economico(proxies_ejemplo)
factores_sociales = fiesta.calcular_factor_social(proxies_ejemplo)
factores_clima = fiesta.calcular_factor_clima(proxies_ejemplo)

print(f"Riesgo de salud: {factores_salud['riesgo_salud']*100:.1f}%")
print(f"Riesgo económico: {factores_economicos['riesgo_economico']*100:.1f}%")
print(f"Factor social: {factores_sociales['factor_fiesta']*100:.1f}%")
print(f"Riesgo climático: {factores_clima['factor_riesgo']*100:.1f}%")

# Resumen final
print("\nRESUMEN FINAL:")
print("=" * 50)
print(f"Decisión: {resultado['decision'].replace('_', ' ')}")
print(f"Confianza: {max(resultado['valores_esperados'])/20*100:.1f}%")
print("=" * 50)
```

    
    === ANÁLISIS AVANZADO DE DECISIÓN PARA FIESTA ===
    
    CONDICIONES BÁSICAS:
    Hora actual: 23:30
    Hora prevista regreso: 02:30
    Nivel de alcohol: 40%
    Cantidad de amigos: 100%
    ¿Ya cenaste?: Sí
    Trabajo mañana: No
    
    FACTORES SOCIALES:
    Presión social: 100%
    Ocasión especial: No
    FOMO: 100%
    
    FACTORES AMBIENTALES:
    Música agradable: 100%
    Lugar agradable: 100%
    Lluvia: 10%
    Temperatura inadecuada: 90%
    Distancia al lugar: 10%
    
    FACTORES DE SALUD:
    Estado de salud: 90%
    Medicamentos: No
    Calidad de comida: 100%
    Deuda de sueño: 10%
    
    FACTORES ECONÓMICOS:
    Gasto esperado: 10%
    Presupuesto disponible: 100%
    Costo transporte: 10%
    Cover: 10%
    
    DECISIÓN SUGERIDA: Salir Sin Preocupaciones
    
    EXPLICACIÓN: Hay muchos amigos | Ambiente muy favorable | Alta presión social
    
    PROBABILIDADES DE CADA ESCENARIO:
    Gran Fiesta Sin Excesos         37.5% ██████████████
    Gran Fiesta Muy Ebrio           13.3% █████
    Mala Fiesta Muy Ebrio            4.7% █
    Fiesta Normal Sin Excesos       21.1% ████████
    Gran Fiesta Incumplimiento      11.7% ████
    Mala Fiesta Incumplimiento       4.7% █
    Accidente O Problema Grave       2.3% 
    Fiesta Aburrida                  0.0% 
    Problema Salud                   2.3% 
    Problema Economico               2.3% 
    
    VALORES ESPERADOS POR DECISIÓN:
    No Salir                         0.00
    Salir Sin Tomar                  8.33
    Salir Y Tomar Moderado          12.85
    Salir Sin Preocupaciones        14.26
    
    FACTORES CRÍTICOS:
    Riesgo de salud: 6.0%
    Riesgo económico: 30.0%
    Factor social: 70.0%
    Riesgo climático: 34.0%
    
    RESUMEN FINAL:
    ==================================================
    Decisión: Salir Sin Preocupaciones
    Confianza: 71.3%
    ==================================================


### Salir y Tomar Moderadamente


```python
fiesta = FiestaDecisionAvanzada()

proxies_ejemplo = {
    # Factores originales
    'musica_agradable': 1,
    'lugar_agradable': 1,
    'nivel_alcohol': 0.3,
    'cantidad_amigos': 1,
    'nivel_cansancio': 0.1,
    'trabajo_mañana': False,
    'transporte_seguro': True,
    'ya_cenaste': False,
    'hora': datetime.strptime("23:30", "%H:%M").time(),
    'hora_regreso': datetime.strptime("02:30", "%H:%M").time(),
    
    # Nuevos factores sociales
    'presion_social': 0.3,
    'ocasion_especial': False,
    'fomo': 0.7,
    
    # Factores climáticos
    'lluvia': 0.2,
    'temperatura_inadecuada': 0.3,
    'distancia_venue': 0.4,
    
    # Factores de salud
    'estado_salud': 0.9,
    'medicamentos': False,
    'calidad_comida': 0.6,
    'deuda_sueño': 0.3,
    
    # Factores económicos
    'gasto_esperado': 0.3,
    'presupuesto': 0.7,
    'costo_transporte': 0.2,
    'cover': 0.1
}

resultado = fiesta.sugerir_decision(proxies_ejemplo)

print("\n=== ANÁLISIS AVANZADO DE DECISIÓN PARA FIESTA ===")
print("\nCONDICIONES BÁSICAS:")
print(f"Hora actual: {proxies_ejemplo['hora'].strftime('%H:%M')}")
print(f"Hora prevista regreso: {proxies_ejemplo['hora_regreso'].strftime('%H:%M')}")
print(f"Nivel de alcohol: {proxies_ejemplo['nivel_alcohol']*100:.0f}%")
print(f"Cantidad de amigos: {proxies_ejemplo['cantidad_amigos']*100:.0f}%")
print(f"¿Ya cenaste?: {'Sí' if proxies_ejemplo['ya_cenaste'] else 'No'}")
print(f"Trabajo mañana: {'Sí' if proxies_ejemplo['trabajo_mañana'] else 'No'}")

print("\nFACTORES SOCIALES:")
print(f"Presión social: {proxies_ejemplo['presion_social']*100:.0f}%")
print(f"Ocasión especial: {'Sí' if proxies_ejemplo['ocasion_especial'] else 'No'}")
print(f"FOMO: {proxies_ejemplo['fomo']*100:.0f}%")

print("\nFACTORES AMBIENTALES:")
print(f"Música agradable: {proxies_ejemplo['musica_agradable']*100:.0f}%")
print(f"Lugar agradable: {proxies_ejemplo['lugar_agradable']*100:.0f}%")
print(f"Lluvia: {proxies_ejemplo['lluvia']*100:.0f}%")
print(f"Temperatura inadecuada: {proxies_ejemplo['temperatura_inadecuada']*100:.0f}%")
print(f"Distancia al lugar: {proxies_ejemplo['distancia_venue']*100:.0f}%")

print("\nFACTORES DE SALUD:")
print(f"Estado de salud: {proxies_ejemplo['estado_salud']*100:.0f}%")
print(f"Medicamentos: {'Sí' if proxies_ejemplo['medicamentos'] else 'No'}")
print(f"Calidad de comida: {proxies_ejemplo['calidad_comida']*100:.0f}%")
print(f"Deuda de sueño: {proxies_ejemplo['deuda_sueño']*100:.0f}%")

print("\nFACTORES ECONÓMICOS:")
print(f"Gasto esperado: {proxies_ejemplo['gasto_esperado']*100:.0f}%")
print(f"Presupuesto disponible: {proxies_ejemplo['presupuesto']*100:.0f}%")
print(f"Costo transporte: {proxies_ejemplo['costo_transporte']*100:.0f}%")
print(f"Cover: {proxies_ejemplo['cover']*100:.0f}%")

print("\nDECISIÓN SUGERIDA:", resultado['decision'].replace('_', ' '))
print("\nEXPLICACIÓN:", resultado['explicacion'])

if resultado['advertencias']:
    print("\nADVERTENCIAS:")
    for adv in resultado['advertencias']:
        print(f"⚠️ {adv}")

if resultado['restricciones']:
    print("\nRESTRICCIONES CRÍTICAS:")
    for res in resultado['restricciones']:
        print(f"🚫 {res}")

print("\nPROBABILIDADES DE CADA ESCENARIO:")
for estado, prob in zip(fiesta.estados, resultado.get('probabilidades', fiesta.p_base)):
    estado_fmt = estado.replace('_', ' ').ljust(30)
    prob_pct = f"{(prob*100):5.1f}%"
    barra = "█" * int(prob * 40)
    print(f"{estado_fmt} {prob_pct} {barra}")

print("\nVALORES ESPERADOS POR DECISIÓN:")
valores = resultado.get('valores_esperados', np.zeros(len(fiesta.decisiones)))
for decision, valor in zip(fiesta.decisiones, valores):
    decision_fmt = decision.replace('_', ' ').ljust(30)
    valor_fmt = f"{valor:6.2f}"
    print(f"{decision_fmt} {valor_fmt}")

# Análisis de factores críticos
print("\nFACTORES CRÍTICOS:")
factores_salud = fiesta.calcular_factor_salud(proxies_ejemplo)
factores_economicos = fiesta.calcular_factor_economico(proxies_ejemplo)
factores_sociales = fiesta.calcular_factor_social(proxies_ejemplo)
factores_clima = fiesta.calcular_factor_clima(proxies_ejemplo)

print(f"Riesgo de salud: {factores_salud['riesgo_salud']*100:.1f}%")
print(f"Riesgo económico: {factores_economicos['riesgo_economico']*100:.1f}%")
print(f"Factor social: {factores_sociales['factor_fiesta']*100:.1f}%")
print(f"Riesgo climático: {factores_clima['factor_riesgo']*100:.1f}%")

# Resumen final
print("\nRESUMEN FINAL:")
print("=" * 50)
print(f"Decisión: {resultado['decision'].replace('_', ' ')}")
print(f"Confianza: {max(resultado['valores_esperados'])/20*100:.1f}%")
print("=" * 50)
```

    
    === ANÁLISIS AVANZADO DE DECISIÓN PARA FIESTA ===
    
    CONDICIONES BÁSICAS:
    Hora actual: 23:30
    Hora prevista regreso: 02:30
    Nivel de alcohol: 30%
    Cantidad de amigos: 100%
    ¿Ya cenaste?: No
    Trabajo mañana: No
    
    FACTORES SOCIALES:
    Presión social: 30%
    Ocasión especial: No
    FOMO: 70%
    
    FACTORES AMBIENTALES:
    Música agradable: 100%
    Lugar agradable: 100%
    Lluvia: 20%
    Temperatura inadecuada: 30%
    Distancia al lugar: 40%
    
    FACTORES DE SALUD:
    Estado de salud: 90%
    Medicamentos: No
    Calidad de comida: 60%
    Deuda de sueño: 30%
    
    FACTORES ECONÓMICOS:
    Gasto esperado: 30%
    Presupuesto disponible: 70%
    Costo transporte: 20%
    Cover: 10%
    
    DECISIÓN SUGERIDA: Salir Y Tomar Moderado
    
    EXPLICACIÓN: Hay muchos amigos | Ambiente muy favorable | Alto riesgo económico
    
    ADVERTENCIAS:
    ⚠️ Gastos cercanos al límite del presupuesto
    
    PROBABILIDADES DE CADA ESCENARIO:
    Gran Fiesta Sin Excesos         28.3% ███████████
    Gran Fiesta Muy Ebrio            8.7% ███
    Mala Fiesta Muy Ebrio            4.3% █
    Fiesta Normal Sin Excesos       19.6% ███████
    Gran Fiesta Incumplimiento      10.9% ████
    Mala Fiesta Incumplimiento       4.3% █
    Accidente O Problema Grave       2.2% 
    Fiesta Aburrida                  0.0% 
    Problema Salud                   2.2% 
    Problema Economico              19.6% ███████
    
    VALORES ESPERADOS POR DECISIÓN:
    No Salir                         0.00
    Salir Sin Tomar                  6.00
    Salir Y Tomar Moderado           9.15
    Salir Sin Preocupaciones         7.65
    
    FACTORES CRÍTICOS:
    Riesgo de salud: 14.0%
    Riesgo económico: 85.7%
    Factor social: 33.0%
    Riesgo climático: 29.0%
    
    RESUMEN FINAL:
    ==================================================
    Decisión: Salir Y Tomar Moderado
    Confianza: 45.8%
    ==================================================


### Salir Sin Tomar


```python

proxies_ejemplo = {
    # Factores originales
    'musica_agradable': 1,
    'lugar_agradable': 1,
    'nivel_alcohol': 0.8,
    'cantidad_amigos': 1,
    'nivel_cansancio': 0.1,
    'trabajo_mañana': True,
    'transporte_seguro': True,
    'ya_cenaste': False,
    'hora': datetime.strptime("23:30", "%H:%M").time(),
    'hora_regreso': datetime.strptime("02:30", "%H:%M").time(),
    
    # Nuevos factores sociales
    'presion_social': 0.3,
    'ocasion_especial': False,
    'fomo': 0.7,
    
    # Factores climáticos
    'lluvia': 0.2,
    'temperatura_inadecuada': 0.3,
    'distancia_venue': 0.4,
    
    # Factores de salud
    'estado_salud': 0.9,
    'medicamentos': False,
    'calidad_comida': 0.6,
    'deuda_sueño': 0.3,
    
    # Factores económicos
    'gasto_esperado': 0.3,
    'presupuesto': 0.7,
    'costo_transporte': 0.2,
    'cover': 0.1
}

resultado = fiesta.sugerir_decision(proxies_ejemplo)

print("\n=== ANÁLISIS AVANZADO DE DECISIÓN PARA FIESTA ===")
print("\nCONDICIONES BÁSICAS:")
print(f"Hora actual: {proxies_ejemplo['hora'].strftime('%H:%M')}")
print(f"Hora prevista regreso: {proxies_ejemplo['hora_regreso'].strftime('%H:%M')}")
print(f"Nivel de alcohol: {proxies_ejemplo['nivel_alcohol']*100:.0f}%")
print(f"Cantidad de amigos: {proxies_ejemplo['cantidad_amigos']*100:.0f}%")
print(f"¿Ya cenaste?: {'Sí' if proxies_ejemplo['ya_cenaste'] else 'No'}")
print(f"Trabajo mañana: {'Sí' if proxies_ejemplo['trabajo_mañana'] else 'No'}")

print("\nFACTORES SOCIALES:")
print(f"Presión social: {proxies_ejemplo['presion_social']*100:.0f}%")
print(f"Ocasión especial: {'Sí' if proxies_ejemplo['ocasion_especial'] else 'No'}")
print(f"FOMO: {proxies_ejemplo['fomo']*100:.0f}%")

print("\nFACTORES AMBIENTALES:")
print(f"Música agradable: {proxies_ejemplo['musica_agradable']*100:.0f}%")
print(f"Lugar agradable: {proxies_ejemplo['lugar_agradable']*100:.0f}%")
print(f"Lluvia: {proxies_ejemplo['lluvia']*100:.0f}%")
print(f"Temperatura inadecuada: {proxies_ejemplo['temperatura_inadecuada']*100:.0f}%")
print(f"Distancia al lugar: {proxies_ejemplo['distancia_venue']*100:.0f}%")

print("\nFACTORES DE SALUD:")
print(f"Estado de salud: {proxies_ejemplo['estado_salud']*100:.0f}%")
print(f"Medicamentos: {'Sí' if proxies_ejemplo['medicamentos'] else 'No'}")
print(f"Calidad de comida: {proxies_ejemplo['calidad_comida']*100:.0f}%")
print(f"Deuda de sueño: {proxies_ejemplo['deuda_sueño']*100:.0f}%")

print("\nFACTORES ECONÓMICOS:")
print(f"Gasto esperado: {proxies_ejemplo['gasto_esperado']*100:.0f}%")
print(f"Presupuesto disponible: {proxies_ejemplo['presupuesto']*100:.0f}%")
print(f"Costo transporte: {proxies_ejemplo['costo_transporte']*100:.0f}%")
print(f"Cover: {proxies_ejemplo['cover']*100:.0f}%")

print("\nDECISIÓN SUGERIDA:", resultado['decision'].replace('_', ' '))
print("\nEXPLICACIÓN:", resultado['explicacion'])

if resultado['advertencias']:
    print("\nADVERTENCIAS:")
    for adv in resultado['advertencias']:
        print(f"⚠️ {adv}")

if resultado['restricciones']:
    print("\nRESTRICCIONES CRÍTICAS:")
    for res in resultado['restricciones']:
        print(f"🚫 {res}")

print("\nPROBABILIDADES DE CADA ESCENARIO:")
for estado, prob in zip(fiesta.estados, resultado.get('probabilidades', fiesta.p_base)):
    estado_fmt = estado.replace('_', ' ').ljust(30)
    prob_pct = f"{(prob*100):5.1f}%"
    barra = "█" * int(prob * 40)
    print(f"{estado_fmt} {prob_pct} {barra}")

print("\nVALORES ESPERADOS POR DECISIÓN:")
valores = resultado.get('valores_esperados', np.zeros(len(fiesta.decisiones)))
for decision, valor in zip(fiesta.decisiones, valores):
    decision_fmt = decision.replace('_', ' ').ljust(30)
    valor_fmt = f"{valor:6.2f}"
    print(f"{decision_fmt} {valor_fmt}")

# Análisis de factores críticos
print("\nFACTORES CRÍTICOS:")
factores_salud = fiesta.calcular_factor_salud(proxies_ejemplo)
factores_economicos = fiesta.calcular_factor_economico(proxies_ejemplo)
factores_sociales = fiesta.calcular_factor_social(proxies_ejemplo)
factores_clima = fiesta.calcular_factor_clima(proxies_ejemplo)

print(f"Riesgo de salud: {factores_salud['riesgo_salud']*100:.1f}%")
print(f"Riesgo económico: {factores_economicos['riesgo_economico']*100:.1f}%")
print(f"Factor social: {factores_sociales['factor_fiesta']*100:.1f}%")
print(f"Riesgo climático: {factores_clima['factor_riesgo']*100:.1f}%")

# Resumen final
print("\nRESUMEN FINAL:")
print("=" * 50)
print(f"Decisión: {resultado['decision'].replace('_', ' ')}")
print(f"Confianza: {max(resultado['valores_esperados'])/20*100:.1f}%")
print("=" * 50)
```

    
    === ANÁLISIS AVANZADO DE DECISIÓN PARA FIESTA ===
    
    CONDICIONES BÁSICAS:
    Hora actual: 23:30
    Hora prevista regreso: 02:30
    Nivel de alcohol: 80%
    Cantidad de amigos: 100%
    ¿Ya cenaste?: No
    Trabajo mañana: Sí
    
    FACTORES SOCIALES:
    Presión social: 30%
    Ocasión especial: No
    FOMO: 70%
    
    FACTORES AMBIENTALES:
    Música agradable: 100%
    Lugar agradable: 100%
    Lluvia: 20%
    Temperatura inadecuada: 30%
    Distancia al lugar: 40%
    
    FACTORES DE SALUD:
    Estado de salud: 90%
    Medicamentos: No
    Calidad de comida: 60%
    Deuda de sueño: 30%
    
    FACTORES ECONÓMICOS:
    Gasto esperado: 30%
    Presupuesto disponible: 70%
    Costo transporte: 20%
    Cover: 10%
    
    DECISIÓN SUGERIDA: Salir Sin Tomar
    
    EXPLICACIÓN: Hay muchos amigos | Ambiente muy favorable | Alto riesgo económico
    
    ADVERTENCIAS:
    ⚠️ Gastos cercanos al límite del presupuesto
    
    RESTRICCIONES CRÍTICAS:
    🚫 Hora de regreso muy tarde con trabajo mañana
    
    PROBABILIDADES DE CADA ESCENARIO:
    Gran Fiesta Sin Excesos         15.2% ██████
    Gran Fiesta Muy Ebrio           12.1% ████
    Mala Fiesta Muy Ebrio            7.6% ███
    Fiesta Normal Sin Excesos       10.6% ████
    Gran Fiesta Incumplimiento      19.7% ███████
    Mala Fiesta Incumplimiento      15.2% ██████
    Accidente O Problema Grave       4.5% █
    Fiesta Aburrida                  0.0% 
    Problema Salud                   1.5% 
    Problema Economico              13.6% █████
    
    VALORES ESPERADOS POR DECISIÓN:
    No Salir                         0.00
    Salir Sin Tomar                  8.61
    Salir Y Tomar Moderado           1.41
    Salir Sin Preocupaciones        -0.05
    
    FACTORES CRÍTICOS:
    Riesgo de salud: 14.0%
    Riesgo económico: 85.7%
    Factor social: 33.0%
    Riesgo climático: 29.0%
    
    RESUMEN FINAL:
    ==================================================
    Decisión: Salir Sin Tomar
    Confianza: 43.0%
    ==================================================

