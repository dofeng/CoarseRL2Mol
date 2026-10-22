"""Compatibility entry point for the SQLite-backed reaction network dashboard."""
from dashboard_app import DashboardHandler, DashboardServer, main, start_dashboard

__all__ = ["DashboardHandler", "DashboardServer", "main", "start_dashboard"]

if __name__ == "__main__":
    main()
