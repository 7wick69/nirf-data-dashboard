from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_CANDIDATES = [ROOT / 'data' / 'clean' / 'data_nirf_clean2.csv', ROOT / 'data' / 'data_nirf_clean2.csv']
DATA = next((candidate for candidate in DATA_CANDIDATES if candidate.exists()), None)
sys.path.insert(0, str(ROOT))

from chatbot.analytics_engine import NIRFAnalyticsEngine
from chatbot.visualization_engine import NIRFVisualizationEngine
from chatbot.chart_specs import normalize_chart_type
from chatbot.intelligence import QuestionLearner


@pytest.fixture(scope='module')
def engine():
    if DATA is None:
        pytest.skip('Place data_nirf_clean2.csv in data/clean/ or data/.')
    return NIRFAnalyticsEngine(pd.read_csv(DATA))


@pytest.fixture(scope='module')
def renderer():
    return NIRFVisualizationEngine()


def run_operation(engine, operation, intent, **fields):
    query = {'intent': intent, 'category': 'Engineering', 'operation': operation, **fields}
    return engine.execute({'status': 'ready', 'query': query, 'institution_keys': []})


def test_ts04_mean_score_by_year(engine, renderer):
    result = run_operation(engine, 'mean_score_by_year', 'trend_analysis', start_year=2017, end_year=2025, metric='score')
    assert result['status'] == 'success'
    assert result['operation'] == 'mean_score_by_year'
    assert result['chart']['x'] == 'year'
    assert result['chart']['y'] == 'mean_score'
    assert len(result['data']) == 9
    assert all(row['institution_count'] > 0 for row in result['data'])
    chart_type = normalize_chart_type('line_trend', result['intent'], 'year', 'mean_score')
    rendered = renderer.create_figure(result, chart_type)
    assert rendered['rendered_type'] == 'line'


def test_ts05_top10_consistency(engine, renderer):
    result = run_operation(engine, 'top10_consistency', 'trend_analysis', start_year=2017, end_year=2025)
    assert result['status'] == 'success'
    assert result['operation'] == 'top10_consistency'
    assert result['statistics']['years_observed'] == 9
    assert result['statistics']['minimum_years_for_consistency'] == 6
    assert len(result['data']) > 0
    year_cols = [str(y) for y in range(2017, 2026)]
    assert all(col in result['data'][0] for col in year_cols)
    assert all(row['top10_year_count'] >= 6 for row in result['data'])
    chart_type = normalize_chart_type('membership_heatmap', result['intent'], 'year', 'top10_year_count')
    rendered = renderer.create_figure(result, chart_type)
    assert rendered['rendered_type'] == 'heatmap'


def test_geo02_state_distribution_by_year(engine, renderer):
    result = engine.execute({'status': 'ready', 'query': {
        'intent': 'geographical_analysis', 'category': 'Overall',
        'operation': 'state_distribution_by_year', 'start_year': 2017,
        'end_year': 2025, 'rank_limit': 100, 'state_limit': 20,
    }, 'institution_keys': []})
    assert result['status'] == 'success'
    assert result['operation'] == 'state_distribution_by_year'
    assert len(result['data']) > 0
    assert '2025' in result['data'][0]
    assert '2017' in result['data'][0]
    chart_type = normalize_chart_type('state_year_heatmap', result['intent'], 'year', 'institution_count')
    rendered = renderer.create_figure(result, chart_type)
    assert rendered['rendered_type'] == 'heatmap'


def test_geo03_state_representation_change(engine, renderer):
    result = engine.execute({'status': 'ready', 'query': {
        'intent': 'geographical_analysis', 'category': 'Overall',
        'operation': 'state_representation_change', 'start_year': 2017,
        'end_year': 2025, 'rank_limit': 100,
    }, 'institution_keys': []})
    assert result['status'] == 'success'
    assert result['operation'] == 'state_representation_change'
    assert {'state', 'start_count', 'end_count', 'share_change_pp'} <= set(result['data'][0])
    chart_type = normalize_chart_type('state_change_bar', result['intent'], 'state', 'share_change_pp')
    rendered = renderer.create_figure(result, chart_type)
    assert rendered['rendered_type'] == 'diverging_bar'


def test_geo05_new_entries_by_state(engine, renderer):
    result = run_operation(engine, 'new_entries_by_state', 'geographical_analysis', start_year=2017, end_year=2025)
    assert result['status'] == 'success'
    assert result['operation'] == 'new_entries_by_state'
    assert result['statistics']['new_institutions'] == len(result['details'])
    assert len(result['data']) > 0
    chart_type = normalize_chart_type('state_new_entries_bar', result['intent'], 'state', 'new_institution_count')
    rendered = renderer.create_figure(result, chart_type)
    assert rendered['rendered_type'] == 'horizontal_bar'


def test_unknown_operation_falls_back_to_existing_intent(engine):
    result = engine.execute({'status': 'ready', 'query': {
        'intent': 'ranking', 'category': 'Engineering', 'year': 2025,
        'rank_limit': 10, 'states': [], 'institutions': [],
        'operation': 'not_an_operation',
    }, 'institution_keys': []})
    assert result['status'] == 'success'
    assert len(result['data']) == 10


def test_geo05_does_not_infer_perception_from_word_produced():
    prediction = QuestionLearner().predict(
        "Which states have produced the largest number of newly appearing institutions in the rankings?"
    )
    assert prediction["operation"] == "new_entries_by_state"
    assert "PR" not in prediction["slots"].get("metrics", [])
