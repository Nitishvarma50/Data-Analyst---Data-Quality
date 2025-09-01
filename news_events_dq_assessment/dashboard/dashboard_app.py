"""
News Events Data Quality Assessment Dashboard
Interactive dashboard for monitoring and visualizing data quality metrics
"""

import dash
from dash import dcc, html, Input, Output, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os
from typing import Dict, List, Optional, Any

# Add src to path for database imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from src.database.database_manager import DatabaseManager
except ImportError:
    print("Warning: Could not import DatabaseManager. Using mock data.")
    DatabaseManager = None

class DataQualityDashboard:
    """
    Main dashboard class for data quality visualization
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.db_manager = DatabaseManager(config_path) if DatabaseManager else None
        self.app = dash.Dash(
            __name__, 
            external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
            meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
        )
        self.app.title = "News Events Data Quality Dashboard"
        
        # Initialize dashboard
        self._setup_layout()
        self._setup_callbacks()
        
    def _get_mock_data(self):
        """Generate mock data if database is not available"""
        return {
            'quality_summary': {
                'assessment_date': datetime.now(),
                'total_records': 4992,
                'sample_size': 3425,
                'completeness_score': 76.04,
                'uniqueness_score': 99.32,
                'validity_score': 95.81,
                'consistency_score': 65.91,
                'timeliness_score': 47.64,
                'overall_score': 79.86
            },
            'database_stats': {
                'events': 3425,
                'companies': 1245,
                'news_articles': 2156,
                'quality_assessments': 3,
                'cleaning_operations': 5
            },
            'events_by_category': pd.DataFrame({
                'category': ['receives_award', 'launches', 'hires', 'receives_financing', 'is_developing'],
                'count': [1456, 892, 534, 378, 165],
                'avg_confidence': [0.85, 0.78, 0.82, 0.91, 0.76]
            }),
            'quality_trends': pd.DataFrame({
                'assessment_date': pd.date_range('2025-08-01', periods=10, freq='3D'),
                'overall_score': [72.5, 74.2, 76.8, 78.1, 79.86, 80.2, 81.5, 82.1, 83.0, 84.2],
                'completeness_score': [70, 72, 74, 75, 76, 77, 78, 79, 80, 81],
                'consistency_score': [60, 62, 63, 64, 66, 67, 68, 69, 70, 71]
            })
        }
    
    def _fetch_dashboard_data(self):
        """Fetch data from database or use mock data"""
        try:
            if self.db_manager:
                # Try to fetch real data
                data = {
                    'quality_summary': self.db_manager.get_quality_summary(),
                    'database_stats': self.db_manager.get_database_stats(),
                    'quality_trends': self.db_manager.get_quality_trends(30),
                    'freshness_report': self.db_manager.get_data_freshness_report()
                }
                
                # Get additional analysis
                quality_queries = self.db_manager.execute_quality_queries()
                data.update(quality_queries)
                
                return data
            else:
                return self._get_mock_data()
                
        except Exception as e:
            print(f"Error fetching data: {e}")
            return self._get_mock_data()
    
    def _setup_layout(self):
        """Setup the dashboard layout"""
        
        # Fetch initial data
        data = self._fetch_dashboard_data()
        quality_summary = data.get('quality_summary', {})
        
        self.app.layout = dbc.Container([
            
            # Header
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H1("News Events Data Quality Dashboard", 
                               className="text-center mb-0"),
                        html.P(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                              className="text-center text-muted"),
                        html.Hr()
                    ])
                ])
            ]),
            
            # Quality Score Cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Overall Quality Score", className="card-title"),
                            html.H2(f"{quality_summary.get('overall_score', 0):.1f}/100", 
                                   className="text-primary"),
                            html.P("Based on 5 quality dimensions", className="card-text")
                        ])
                    ], color="primary", outline=True)
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Records Processed", className="card-title"),
                            html.H2(f"{quality_summary.get('total_records', 0):,}", 
                                   className="text-info"),
                            html.P(f"Clean: {quality_summary.get('sample_size', 0):,}", 
                                  className="card-text")
                        ])
                    ], color="info", outline=True)
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Data Freshness", className="card-title"),
                            html.H2(f"{quality_summary.get('timeliness_score', 0):.1f}%", 
                                   className="text-warning"),
                            html.P("Timeliness score", className="card-text")
                        ])
                    ], color="warning", outline=True)
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Data Consistency", className="card-title"),
                            html.H2(f"{quality_summary.get('consistency_score', 0):.1f}%", 
                                   className="text-danger" if quality_summary.get('consistency_score', 0) < 70 else "text-success"),
                            html.P("Location & category standardization", className="card-text")
                        ])
                    ], color="danger" if quality_summary.get('consistency_score', 0) < 70 else "success", outline=True)
                ], width=3)
            ], className="mb-4"),
            
            # Quality Dimensions Chart
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Quality Dimensions Breakdown")),
                        dbc.CardBody([
                            dcc.Graph(id="quality-dimensions-chart")
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Quality Trends Over Time")),
                        dbc.CardBody([
                            dcc.Graph(id="quality-trends-chart")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Events Analysis
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Events by Category")),
                        dbc.CardBody([
                            dcc.Graph(id="events-category-chart")
                        ])
                    ])
                ], width=8),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Database Statistics")),
                        dbc.CardBody([
                            html.Div(id="database-stats-table")
                        ])
                    ])
                ], width=4)
            ], className="mb-4"),
            
            # Data Quality Issues
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4("Data Quality Issues & Recommendations", className="mb-0")
                        ]),
                        dbc.CardBody([
                            html.Div(id="quality-issues-panel")
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Recent High-Confidence Events
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Recent High-Confidence Events")),
                        dbc.CardBody([
                            html.Div(id="recent-events-table")
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Control Panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Dashboard Controls")),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("Refresh Data", id="refresh-btn", 
                                             color="primary", className="me-2"),
                                    dbc.Button("Export Report", id="export-btn", 
                                             color="secondary", className="me-2"),
                                    dbc.Button("View Database", id="database-btn", 
                                             color="info")
                                ], width=6),
                                dbc.Col([
                                    html.P("Dashboard Status: ", className="mb-1"),
                                    dbc.Badge("Active", color="success", id="status-badge"),
                                    html.P(id="last-refresh", className="text-muted small mt-2")
                                ], width=6)
                            ])
                        ])
                    ])
                ])
            ])
            
        ], fluid=True, className="py-3")
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks for interactivity"""
        
        @self.app.callback(
            [Output("quality-dimensions-chart", "figure"),
             Output("quality-trends-chart", "figure"),
             Output("events-category-chart", "figure"),
             Output("database-stats-table", "children"),
             Output("quality-issues-panel", "children"),
             Output("recent-events-table", "children"),
             Output("last-refresh", "children")],
            [Input("refresh-btn", "n_clicks")]
        )
        def update_dashboard(n_clicks):
            """Update all dashboard components"""
            
            # Fetch latest data
            data = self._fetch_dashboard_data()
            
            # Quality dimensions radar chart
            quality_summary = data.get('quality_summary', {})
            dimensions_fig = self._create_quality_dimensions_chart(quality_summary)
            
            # Quality trends line chart
            trends_data = data.get('quality_trends', pd.DataFrame())
            trends_fig = self._create_quality_trends_chart(trends_data)
            
            # Events by category
            category_data = data.get('events_by_category', pd.DataFrame())
            category_fig = self._create_events_category_chart(category_data)
            
            # Database stats table
            db_stats = data.get('database_stats', {})
            stats_table = self._create_database_stats_table(db_stats)
            
            # Quality issues panel
            issues_panel = self._create_quality_issues_panel(quality_summary)
            
            # Recent events table
            recent_events = data.get('recent_high_confidence', pd.DataFrame())
            events_table = self._create_recent_events_table(recent_events)
            
            # Last refresh timestamp
            refresh_time = f"Last refresh: {datetime.now().strftime('%H:%M:%S')}"
            
            return (dimensions_fig, trends_fig, category_fig, stats_table, 
                   issues_panel, events_table, refresh_time)
    
    def _create_quality_dimensions_chart(self, quality_summary: dict) -> go.Figure:
        """Create radar chart for quality dimensions"""
        
        dimensions = ['Completeness', 'Uniqueness', 'Validity', 'Consistency', 'Timeliness']
        scores = [
            quality_summary.get('completeness_score', 0),
            quality_summary.get('uniqueness_score', 0),
            quality_summary.get('validity_score', 0),
            quality_summary.get('consistency_score', 0),
            quality_summary.get('timeliness_score', 0)
        ]
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=dimensions,
            fill='toself',
            name='Current Scores',
            line_color='rgb(0, 123, 255)',
            fillcolor='rgba(0, 123, 255, 0.3)'
        ))
        
        # Add threshold line
        threshold_scores = [75] * len(dimensions)  # 75% threshold
        fig.add_trace(go.Scatterpolar(
            r=threshold_scores,
            theta=dimensions,
            fill=None,
            name='Target (75%)',
            line=dict(color='red', dash='dash'),
            showlegend=True
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            title="Data Quality Dimensions",
            height=400
        )
        
        return fig
    
    def _create_quality_trends_chart(self, trends_data: pd.DataFrame) -> go.Figure:
        """Create line chart for quality trends over time"""
        
        if trends_data.empty:
            # Create mock trends data
            dates = pd.date_range(start='2025-08-01', end='2025-09-01', freq='3D')
            trends_data = pd.DataFrame({
                'assessment_date': dates,
                'overall_score': [72 + i*0.5 for i in range(len(dates))],
                'completeness_score': [70 + i*0.4 for i in range(len(dates))],
                'consistency_score': [60 + i*0.6 for i in range(len(dates))]
            })
        
        fig = go.Figure()
        
        # Overall score
        fig.add_trace(go.Scatter(
            x=trends_data['assessment_date'],
            y=trends_data['overall_score'],
            mode='lines+markers',
            name='Overall Score',
            line=dict(color='rgb(0, 123, 255)', width=3),
            marker=dict(size=6)
        ))
        
        # Individual dimensions
        if 'completeness_score' in trends_data.columns:
            fig.add_trace(go.Scatter(
                x=trends_data['assessment_date'],
                y=trends_data['completeness_score'],
                mode='lines',
                name='Completeness',
                line=dict(color='green', dash='dot')
            ))
        
        if 'consistency_score' in trends_data.columns:
            fig.add_trace(go.Scatter(
                x=trends_data['assessment_date'],
                y=trends_data['consistency_score'],
                mode='lines',
                name='Consistency',
                line=dict(color='orange', dash='dot')
            ))
        
        # Add target line
        fig.add_hline(y=75, line_dash="dash", line_color="red", 
                     annotation_text="Target (75%)")
        
        fig.update_layout(
            title="Quality Score Trends",
            xaxis_title="Date",
            yaxis_title="Quality Score (%)",
            yaxis_range=[0, 100],
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def _create_events_category_chart(self, category_data: pd.DataFrame) -> go.Figure:
        """Create bar chart for events by category"""
        
        if category_data.empty:
            # Mock data based on your actual results
            category_data = pd.DataFrame({
                'category': ['receives_award', 'launches', 'hires', 'receives_financing', 'is_developing'],
                'count': [1456, 892, 534, 378, 165],
                'avg_confidence': [0.85, 0.78, 0.82, 0.91, 0.76]
            })
        
        # Create subplot with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Event counts
        fig.add_trace(
            go.Bar(
                x=category_data['category'],
                y=category_data['count'],
                name='Event Count',
                marker_color='lightblue',
                yaxis='y'
            ),
            secondary_y=False
        )
        
        # Average confidence
        fig.add_trace(
            go.Scatter(
                x=category_data['category'],
                y=category_data['avg_confidence'],
                mode='markers+lines',
                name='Avg Confidence',
                marker=dict(color='red', size=8),
                line=dict(color='red'),
                yaxis='y2'
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_xaxes(title_text="Event Category")
        fig.update_yaxes(title_text="Number of Events", secondary_y=False)
        fig.update_yaxes(title_text="Average Confidence", secondary_y=True, range=[0, 1])
        
        fig.update_layout(
            title="Events Distribution by Category",
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def _create_database_stats_table(self, db_stats: dict) -> dbc.Table:
        """Create table showing database statistics"""
        
        if not db_stats:
            db_stats = {
                'events': 3425,
                'companies': 1245,
                'news_articles': 2156,
                'event_companies': 3425,
                'quality_assessments': 3,
                'cleaning_operations': 5
            }
        
        # Create table rows
        table_rows = []
        for table_name, count in db_stats.items():
            formatted_name = table_name.replace('_', ' ').title()
            table_rows.append(
                html.Tr([
                    html.Td(formatted_name),
                    html.Td(f"{count:,}", className="text-end")
                ])
            )
        
        return dbc.Table([
            html.Thead([
                html.Tr([
                    html.Th("Table"),
                    html.Th("Records", className="text-end")
                ])
            ]),
            html.Tbody(table_rows)
        ], bordered=True, hover=True, size="sm")
    
    def _create_quality_issues_panel(self, quality_summary: dict) -> html.Div:
        """Create panel showing quality issues and recommendations"""
        
        issues = []
        
        # Check each dimension against thresholds
        dimensions = [
            ('completeness_score', 'Completeness', 75, 'Missing data in critical fields'),
            ('uniqueness_score', 'Uniqueness', 95, 'Duplicate records detected'),
            ('validity_score', 'Validity', 90, 'Format validation issues'),
            ('consistency_score', 'Consistency', 75, 'Inconsistent categories/locations'),
            ('timeliness_score', 'Timeliness', 70, 'Stale data detected')
        ]
        
        for score_key, dimension_name, threshold, issue_desc in dimensions:
            score = quality_summary.get(score_key, 0)
            
            if score < threshold:
                severity = "High" if score < threshold - 20 else "Medium"
                color = "danger" if severity == "High" else "warning"
                
                issues.append(
                    dbc.Alert([
                        html.H5(f"{dimension_name}: {score:.1f}%", className="alert-heading"),
                        html.P(f"{issue_desc} (Target: {threshold}%)"),
                        html.Hr(),
                        html.P(f"Priority: {severity}", className="mb-0 small")
                    ], color=color, className="mb-2")
                )
        
        if not issues:
            issues = [
                dbc.Alert([
                    html.H5("All Quality Checks Passed!", className="alert-heading"),
                    html.P("No critical data quality issues detected.")
                ], color="success")
            ]
        
        return html.Div(issues)
    
    def _create_recent_events_table(self, recent_events: pd.DataFrame) -> dbc.Table:
        """Create table showing recent high-confidence events"""
        
        if recent_events.empty:
            # Mock recent events
            recent_events = pd.DataFrame({
                'event_id': ['evt_001', 'evt_002', 'evt_003'],
                'summary': [
                    'TechCorp receives innovation award for AI platform',
                    'StartupXYZ launches new mobile application',
                    'MegaCorp hires new Chief Technology Officer'
                ],
                'category': ['receives_award', 'launches', 'hires'],
                'confidence': [0.95, 0.88, 0.92],
                'company_name': ['TechCorp', 'StartupXYZ', 'MegaCorp']
            })
        
        # Limit to top 10 for display
        recent_events = recent_events.head(10)
        
        table_rows = []
        for _, row in recent_events.iterrows():
            confidence_color = "success" if row['confidence'] > 0.9 else "warning" if row['confidence'] > 0.8 else "secondary"
            
            table_rows.append(
                html.Tr([
                    html.Td(row.get('company_name', 'Unknown')),
                    html.Td(row['summary'][:80] + "..." if len(row['summary']) > 80 else row['summary']),
                    html.Td(row['category'].replace('_', ' ').title()),
                    html.Td([
                        dbc.Badge(f"{row['confidence']:.2f}", color=confidence_color)
                    ], className="text-center")
                ])
            )
        
        return dbc.Table([
            html.Thead([
                html.Tr([
                    html.Th("Company"),
                    html.Th("Event Summary"),
                    html.Th("Category"),
                    html.Th("Confidence", className="text-center")
                ])
            ]),
            html.Tbody(table_rows)
        ], bordered=True, hover=True, size="sm")
    
    def run_server(self, debug: bool = True, host: str = "127.0.0.1", port: int = 8050):
        """Run the dashboard server"""
        print(f"Starting Data Quality Dashboard...")
        print(f"Access dashboard at: http://{host}:{port}")
        print("-" * 50)
        
        self.app.run(debug=debug, host=host, port=port)


def create_dashboard_launcher():
    """Create a simple launcher script"""
    
    launcher_content = '''
"""
Dashboard Launcher Script
Quick way to start the data quality dashboard
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from dashboard.app import DataQualityDashboard

def main():
    print("🚀 Launching News Events Data Quality Dashboard...")
    
    # Check if database exists
    db_path = "news_events_dq.db"
    if not Path(db_path).exists():
        print(f"⚠️  Database not found: {db_path}")
        print("💡 Run the main pipeline first: python main.py --mode database-only --data-path data/processed/cleaned_events.json")
        print("📊 Dashboard will use mock data for demonstration")
    
    # Start dashboard
    dashboard = DataQualityDashboard("config/config.yaml")
    dashboard.run_server(debug=True, host="127.0.0.1", port=8050)

if __name__ == "__main__":
    main()
'''
    
    return launcher_content


def main():
    """Main function for command line usage"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='News Events Data Quality Dashboard')
    parser.add_argument('--host', default='127.0.0.1', help='Dashboard host')
    parser.add_argument('--port', type=int, default=8050, help='Dashboard port')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    # Initialize and run dashboard
    dashboard = DataQualityDashboard(args.config)
    dashboard.run_server(debug=args.debug, host=args.host, port=args.port)


if __name__ == "__main__":
    main()