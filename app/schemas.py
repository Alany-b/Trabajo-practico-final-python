from pydantic import BaseModel, Field
from typing import List, Tuple, Optional

class PredictRequest(BaseModel):
    """
    Esquema de entrada para solicitar la predicción de un partido individual.
    """
    team_local: str = Field(..., description="Nombre de la selección local o Equipo A", example="Argentina")
    team_visitor: str = Field(..., description="Nombre de la selección visitante o Equipo B", example="France")
    is_neutral: bool = Field(True, description="Indica si el partido se juega en cancha neutral (como en el Mundial)")
    use_elo: bool = Field(True, description="Indica si se debe aplicar el ajuste del rating ELO actual a la predicción")


class PredictResponse(BaseModel):
    """
    Esquema de respuesta para la predicción de un partido individual.
    """
    team_local: str
    team_visitor: str
    xg_local: float = Field(..., description="Goles Esperados (xG) calculados para el Equipo A")
    xg_visitor: float = Field(..., description="Goles Esperados (xG) calculados para el Equipo B")
    prob_local: float = Field(..., description="Probabilidad de victoria del Equipo A (rango 0-1)")
    prob_draw: float = Field(..., description="Probabilidad de empate (rango 0-1)")
    prob_visitor: float = Field(..., description="Probabilidad de victoria del Equipo B (rango 0-1)")
    most_probable_score: Tuple[int, int] = Field(..., description="Marcador exacto más probable (goles_A, goles_B)")
    prob_score: float = Field(..., description="Probabilidad asociada al marcador más probable")


class MatchResultInput(BaseModel):
    """
    Esquema de entrada para registrar el resultado real de un partido disputado.
    Sirve para actualizar la base de datos y recalcular ratings.
    """
    team_local: str = Field(..., description="Nombre del equipo local")
    team_visitor: str = Field(..., description="Nombre del equipo visitante")
    goals_local: int = Field(..., description="Goles marcados por el equipo local", ge=0)
    goals_visitor: int = Field(..., description="Goles marcados por el equipo visitante", ge=0)
    is_neutral: bool = Field(True, description="Indica si el partido fue en cancha neutral")
    competition: str = Field("Mundial 2026", description="Nombre de la competición (ej. Mundial 2026, Liga Española)")


class TeamInfo(BaseModel):
    """
    Esquema de información general de un equipo nacional, incluyendo su ranking y rating ELO.
    """
    team: str
    team_group: Optional[str] = None
    confederation: str
    fifa_rank: Optional[int] = None
    elo_rating: float
    coach: Optional[str] = None


class SimulationResponse(BaseModel):
    """
    Esquema de respuesta rápida con los resultados de una única simulación de torneo de simulación directa.
    """
    campeon: str
    subcampeon: str
    tercero: str
    cuarto: str
    semifinalistas: List[str]
    cuartofinalistas: List[str]

class PredictMLResponse(BaseModel):
    """
    Esquema de respuesta para la predicción de Machine Learning (Scikit-Learn).
    """
    team_local: str
    team_visitor: str
    prob_local: float = Field(..., description="Probabilidad de victoria del Equipo A")
    prob_draw: float = Field(..., description="Probabilidad de empate")
    prob_visitor: float = Field(..., description="Probabilidad de victoria del Equipo B")
    predicted_result: str = Field(..., description="Resultado predicho categóricamente (ej. 'Gana Local')")
