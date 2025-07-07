"""
Visualization Agent - Creates interactive dashboards and charts.
"""
import asyncio
import logging
import time
import uuid
from typing import Dict, List, Any, Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core.types import AgentState, FinancialKPI
from services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


class VisualizationAgent:
    """
    Agent specialized in creating interactive visualizations and dashboards
    for financial data analysis.
    """
    
    def __init__(self, vector_service: VectorStoreService):
        self.vector_service = vector_service
        self.agent_type = "visualization"
    
    async def create_visualizations(self, state: AgentState) -> AgentState:
        """Create visualizations based on query results and task type."""
        start_time = time.time()
        
        try:
            if not state.query_results:
                state.error_messages.append("No data available for visualization")
                return state
            
            # Determine visualization type based on task and data
            viz_config = self._determine_visualization_config(state)
            
            # Create appropriate visualizations
            if viz_config['type'] == 'dashboard':
                html_content = await self._create_comprehensive_dashboard(
                    state.query_results, state.target_kpis, viz_config
                )
            elif viz_config['type'] == 'trend':
                html_content = await self._create_trend_analysis(
                    state.query_results, viz_config
                )
            elif viz_config['type'] == 'comparison':
                html_content = await self._create_comparison_charts(
                    state.query_results, viz_config
                )
            else:
                html_content = await self._create_basic_charts(
                    state.query_results, viz_config
                )
            
            # Store visualization
            viz_id = str(uuid.uuid4())
            state.visualizations[viz_id] = html_content
            
            # Add metadata
            state.metadata['visualization_metadata'] = {
                'visualization_id': viz_id,
                'chart_type': viz_config['type'],
                'data_points_visualized': self._count_data_points(state.query_results),
                'interactive_features': viz_config.get('interactive_features', [])
            }
            
            # Log execution
            execution_time = time.time() - start_time
            await self.vector_service.log_agent_execution(
                session_id=state.session_id,
                agent_name=self.agent_type,
                input_data={'kpis': [kpi.name for kpi in state.target_kpis]},
                output_data={'visualization_created': True, 'chart_type': viz_config['type']},
                execution_time=execution_time,
                model_used='non-llm'
            )
            
            logger.info(f"Created {viz_config['type']} visualization in {execution_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Visualization agent failed: {e}")
            state.error_messages.append(f"Visualization creation failed: {str(e)}")
        
        return state
    
    def _determine_visualization_config(self, state: AgentState) -> Dict:
        """Determine the best visualization approach based on data and context."""
        config = {
            'type': 'basic',
            'interactive_features': ['zoom', 'hover'],
            'color_scheme': 'viridis',
            'layout': 'grid'
        }
        
        # Determine visualization type based on task type
        if state.task_type and 'DASHBOARD' in state.task_type.value:
            config['type'] = 'dashboard'
            config['layout'] = 'multi_panel'
        elif state.task_type and 'TREND' in state.task_type.value:
            config['type'] = 'trend'
            config['interactive_features'].append('range_selector')
        elif state.task_type and 'COMPARATIVE' in state.task_type.value:
            config['type'] = 'comparison'
            config['layout'] = 'side_by_side'
        
        # Adjust based on number of KPIs
        if len(state.target_kpis) > 3:
            config['layout'] = 'multi_panel'
            config['type'] = 'dashboard'
        
        # Adjust based on data complexity
        total_data_points = self._count_data_points(state.query_results)
        if total_data_points > 100:
            config['interactive_features'].extend(['filter', 'download'])
        
        return config
    
    async def _create_comprehensive_dashboard(self, query_results: Dict, 
                                           target_kpis: List[FinancialKPI], 
                                           config: Dict) -> str:
        """Create a comprehensive dashboard with multiple panels."""
        try:
            # Calculate grid layout
            num_charts = len(query_results)
            cols = min(3, num_charts)
            rows = (num_charts + cols - 1) // cols
            
            # Create subplots
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=list(query_results.keys()),
                specs=[[{"secondary_y": True} for _ in range(cols)] for _ in range(rows)]
            )
            
            # Add charts for each KPI
            for i, (kpi_name, data) in enumerate(query_results.items()):
                row = (i // cols) + 1
                col = (i % cols) + 1
                
                chart = self._create_kpi_chart(kpi_name, data)
                
                if chart:
                    for trace in chart.data:
                        fig.add_trace(trace, row=row, col=col)
            
            # Update layout
            fig.update_layout(
                title={
                    'text': "Financial KPI Dashboard",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 24}
                },
                showlegend=True,
                height=300 * rows,
                template='plotly_white'
            )
            
            # Add annotations for insights
            annotations = self._generate_chart_annotations(query_results)
            if annotations:
                fig.update_layout(annotations=annotations)
            
            return fig.to_html(
                include_plotlyjs='cdn',
                config={'displayModeBar': True, 'responsive': True}
            )
            
        except Exception as e:
            logger.error(f"Dashboard creation failed: {e}")
            return f"<html><body><h3>Dashboard Error</h3><p>{str(e)}</p></body></html>"
    
    async def _create_trend_analysis(self, query_results: Dict, config: Dict) -> str:
        """Create trend analysis visualizations."""
        try:
            fig = go.Figure()
            
            for kpi_name, data in query_results.items():
                if isinstance(data, list) and data:
                    # Assume time series data
                    df = pd.DataFrame(data)
                    if len(df.columns) >= 2:
                        x_col = df.columns[0]
                        y_col = df.columns[1]
                        
                        fig.add_trace(go.Scatter(
                            x=df[x_col],
                            y=df[y_col],
                            mode='lines+markers',
                            name=kpi_name,
                            line=dict(width=3),
                            marker=dict(size=8)
                        ))
            
            fig.update_layout(
                title="Financial Trend Analysis",
                xaxis_title="Time Period",
                yaxis_title="Value",
                template='plotly_white',
                hovermode='x unified'
            )
            
            # Add range selector
            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=7, label="7d", step="day", stepmode="backward"),
                            dict(count=30, label="30d", step="day", stepmode="backward"),
                            dict(count=90, label="3m", step="day", stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type="date"
                )
            )
            
            return fig.to_html(include_plotlyjs='cdn')
            
        except Exception as e:
            logger.error(f"Trend analysis creation failed: {e}")
            return f"<html><body><h3>Trend Analysis Error</h3><p>{str(e)}</p></body></html>"
    
    async def _create_comparison_charts(self, query_results: Dict, config: Dict) -> str:
        """Create comparison charts for multiple KPIs."""
        try:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=["Current Values", "Comparative Analysis"],
                specs=[[{"type": "bar"}, {"type": "radar"}]]
            )
            
            # Bar chart for current values
            kpi_names = list(query_results.keys())
            values = []
            
            for kpi_name, data in query_results.items():
                if isinstance(data, (int, float)):
                    values.append(data)
                elif isinstance(data, list) and data:
                    values.append(len(data))
                else:
                    values.append(0)
            
            fig.add_trace(
                go.Bar(x=kpi_names, y=values, name="Current Values"),
                row=1, col=1
            )
            
            # Radar chart for comparison
            if len(values) >= 3:
                # Normalize values for radar chart
                max_val = max(values) if values else 1
                normalized_values = [v/max_val * 100 for v in values]
                
                fig.add_trace(
                    go.Scatterpolar(
                        r=normalized_values,
                        theta=kpi_names,
                        fill='toself',
                        name='Performance'
                    ),
                    row=1, col=2
                )
            
            fig.update_layout(
                title="KPI Comparison Analysis",
                template='plotly_white'
            )
            
            return fig.to_html(include_plotlyjs='cdn')
            
        except Exception as e:
            logger.error(f"Comparison chart creation failed: {e}")
            return f"<html><body><h3>Comparison Error</h3><p>{str(e)}</p></body></html>"
    
    async def _create_basic_charts(self, query_results: Dict, config: Dict) -> str:
        """Create basic charts for simple visualizations."""
        try:
            fig = go.Figure()
            
            for kpi_name, data in query_results.items():
                chart = self._create_kpi_chart(kpi_name, data)
                if chart:
                    for trace in chart.data:
                        fig.add_trace(trace)
            
            fig.update_layout(
                title="Financial KPI Analysis",
                template='plotly_white',
                showlegend=True
            )
            
            return fig.to_html(include_plotlyjs='cdn')
            
        except Exception as e:
            logger.error(f"Basic chart creation failed: {e}")
            return f"<html><body><h3>Chart Error</h3><p>{str(e)}</p></body></html>"
    
    def _create_kpi_chart(self, kpi_name: str, data: Any) -> Optional[go.Figure]:
        """Create appropriate chart for a single KPI."""
        try:
            if isinstance(data, (int, float)):
                # Single value - create indicator
                fig = go.Figure(go.Indicator(
                    mode="number",
                    value=data,
                    title={"text": kpi_name},
                    number={'suffix': self._get_kpi_suffix(kpi_name)}
                ))
                return fig
            
            elif isinstance(data, list) and data:
                df = pd.DataFrame(data)
                if len(df.columns) >= 2:
                    # Multi-column data - create bar or line chart
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=df.iloc[:, 0],
                        y=df.iloc[:, 1],
                        name=kpi_name
                    ))
                    return fig
                else:
                    # Single column - create histogram
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=df.iloc[:, 0],
                        name=kpi_name
                    ))
                    return fig
            
            return None
            
        except Exception as e:
            logger.error(f"KPI chart creation failed for {kpi_name}: {e}")
            return None
    
    def _get_kpi_suffix(self, kpi_name: str) -> str:
        """Get appropriate suffix for KPI display."""
        if 'rate' in kpi_name.lower() or 'percentage' in kpi_name.lower():
            return '%'
        elif 'value' in kpi_name.lower() or 'amount' in kpi_name.lower():
            return '$'
        elif 'count' in kpi_name.lower() or 'volume' in kpi_name.lower():
            return ''
        else:
            return ''
    
    def _count_data_points(self, query_results: Dict) -> int:
        """Count total data points in query results."""
        total = 0
        for data in query_results.values():
            if isinstance(data, list):
                total += len(data)
            elif isinstance(data, (int, float, str)) and not isinstance(data, dict):
                total += 1
        return total
    
    def _generate_chart_annotations(self, query_results: Dict) -> List[Dict]:
        """Generate insights annotations for charts."""
        annotations = []
        
        try:
            # Find notable values
            for kpi_name, data in query_results.items():
                if isinstance(data, (int, float)) and data > 1000000:
                    annotations.append({
                        'text': f"{kpi_name}: High value detected",
                        'xref': 'paper', 'yref': 'paper',
                        'x': 0.1, 'y': 0.9 - len(annotations) * 0.1,
                        'showarrow': False,
                        'font': {'color': 'red'}
                    })
        except Exception as e:
            logger.error(f"Annotation generation failed: {e}")
        
        return annotations
