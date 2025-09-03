#!/usr/bin/env python3
# weather/tool.py - Weather tool using wttr.in API

from datetime import datetime
from typing import Dict, Any, Optional

import httpx
from claude_agent_toolkit import BaseTool, tool
from weather import WeatherAPI


class WeatherTool(BaseTool):
    """A comprehensive weather tool providing current conditions and forecasts. Users manage data explicitly."""
    
    def __init__(self):
        super().__init__()
        # Explicit data management - no automatic state management  
        self.recent_queries = []
        self.query_count = 0
        self.favorite_locations = []
        self.last_location = None
        
        # Initialize persistent HTTP client for better performance
        self.client = httpx.AsyncClient(
            base_url="https://wttr.in",
            timeout=httpx.Timeout(30.0),
            headers={'User-Agent': 'Claude-Agent-Toolkit-Weather-Demo/1.0'}
        )
    
    
    def _record_query(self, location: str, query_type: str) -> None:
        """Record a weather query in the history."""
        self.query_count += 1
        self.last_location = location
        self.recent_queries.append({
            "id": self.query_count,
            "location": location,
            "query_type": query_type,
            "timestamp": datetime.now().isoformat()
        })
        # Keep only last 20 queries
        if len(self.recent_queries) > 20:
            self.recent_queries = self.recent_queries[-20:]
    
    @tool(description="Get current weather conditions for a location")
    async def get_current_weather(self, location: str = "") -> Dict[str, Any]:
        """Get current weather conditions for a specific location."""
        self._record_query(location or "current location", "current_weather")
        
        # Use WeatherAPI with our persistent client
        result = await WeatherAPI.get_current_weather(self.client, location)
        
        if result.get("success"):
            loc_info = result.get("location", {})
            current = result.get("current", {})
            
            print(f"\n🌤️  [Weather] Current conditions for {loc_info.get('name', location)}: "
                  f"{current.get('condition')} {current.get('temperature_c')}°C\n")
            
            return {
                "success": True,
                "location": loc_info,
                "current_conditions": current,
                "observation_time": result.get("timestamp"),
                "message": f"Current weather retrieved for {loc_info.get('name', location)}"
            }
        else:
            error_msg = result.get("error", "Unknown error")
            print(f"\n❌ [Weather] {error_msg} for {location}\n")
            return {
                "success": False,
                "error": error_msg,
                "location": location,
                "message": f"Failed to retrieve weather data for {location}"
            }
    
    @tool(description="Get weather forecast for a location (1-3 days)")
    async def get_forecast(self, location: str = "", days: int = 3) -> Dict[str, Any]:
        """Get weather forecast for a specific location."""
        days = max(1, min(days, 3))  # Ensure valid range
        self._record_query(location or "current location", f"forecast_{days}d")
        
        # Use WeatherAPI with our persistent client
        result = await WeatherAPI.get_forecast(self.client, location, days)
        
        if result.get("success"):
            loc_info = result.get("location", {})
            
            print(f"\n🌦️  [Weather] {days}-day forecast for {loc_info.get('name', location)} retrieved\n")
            
            return {
                "success": True,
                "location": loc_info,
                "forecast": result.get("forecast", []),
                "days_requested": days,
                "message": f"{days}-day forecast retrieved for {loc_info.get('name', location)}"
            }
        else:
            error_msg = result.get("error", "Unknown error")
            print(f"\n❌ [Weather] {error_msg} for {location}\n")
            return {
                "success": False,
                "error": error_msg,
                "location": location,
                "days_requested": days,
                "message": f"Failed to retrieve forecast for {location}"
            }
    
    @tool(description="Get a simple weather summary for a location")
    async def get_weather_summary(self, location: str = "") -> Dict[str, Any]:
        """Get a brief weather summary for a location."""
        self._record_query(location or "current location", "summary")
        
        # Use WeatherAPI with our persistent client
        result = await WeatherAPI.get_weather_summary(self.client, location)
        
        if result.get("success"):
            summary = result.get("summary", "")
            print(f"\n📝 [Weather] Summary for {location or 'current location'}: {summary}\n")
            
            return {
                "success": True,
                "location": result.get("location"),
                "summary": summary,
                "message": f"Weather summary retrieved for {location or 'current location'}"
            }
        else:
            error_msg = result.get("error", "Unknown error")
            print(f"\n❌ [Weather] {error_msg} for {location}\n")
            return {
                "success": False,
                "error": error_msg,
                "location": location,
                "message": f"Failed to retrieve weather summary for {location}"
            }
    
    @tool(description="Compare weather between two locations")
    async def compare_weather(self, location1: str, location2: str) -> Dict[str, Any]:
        """Compare current weather conditions between two locations."""
        self._record_query(f"{location1} vs {location2}", "comparison")
        
        # Use the get_current_weather method for both locations
        result1 = await self.get_current_weather(location1)
        result2 = await self.get_current_weather(location2)
        
        if result1.get("success") and result2.get("success"):
            loc1_info = result1.get("location", {})
            loc2_info = result2.get("location", {})
            current1 = result1.get("current_conditions", {})
            current2 = result2.get("current_conditions", {})
            
            print(f"\n🔄 [Weather] Comparing {loc1_info.get('name')} vs {loc2_info.get('name')}\n")
            
            # Calculate differences
            temp_diff_c = current1.get('temperature_c', 0) - current2.get('temperature_c', 0)
            humidity_diff = current1.get('humidity', 0) - current2.get('humidity', 0)
            
            return {
                "success": True,
                "location1": {
                    "info": loc1_info,
                    "conditions": current1
                },
                "location2": {
                    "info": loc2_info,
                    "conditions": current2
                },
                "comparison": {
                    "temperature_difference_c": temp_diff_c,
                    "humidity_difference": humidity_diff,
                    "warmer_location": loc1_info.get('name') if temp_diff_c > 0 else loc2_info.get('name'),
                    "more_humid_location": loc1_info.get('name') if humidity_diff > 0 else loc2_info.get('name')
                },
                "message": f"Weather comparison completed for {loc1_info.get('name')} and {loc2_info.get('name')}"
            }
        else:
            errors = []
            if not result1.get("success"):
                errors.append(f"Location 1 ({location1}): {result1.get('error')}")
            if not result2.get("success"):
                errors.append(f"Location 2 ({location2}): {result2.get('error')}")
            
            return {
                "success": False,
                "errors": errors,
                "location1": location1,
                "location2": location2,
                "message": f"Failed to compare weather between {location1} and {location2}"
            }
    
    @tool(description="Add a location to favorites list")
    async def add_favorite_location(self, location: str, nickname: str = "") -> Dict[str, Any]:
        """Add a location to the favorites list."""
        # Check if already in favorites
        existing = next((fav for fav in self.favorite_locations 
                        if fav["location"].lower() == location.lower()), None)
        
        if existing:
            return {
                "success": False,
                "location": location,
                "message": f"{location} is already in your favorites list",
                "favorites": self.favorite_locations
            }
        
        favorite = {
            "location": location,
            "nickname": nickname or location,
            "added_at": datetime.now().isoformat()
        }
        
        self.favorite_locations.append(favorite)
        
        print(f"\n⭐ [Weather] Added {location} to favorites as '{nickname or location}'\n")
        
        return {
            "success": True,
            "location": location,
            "nickname": nickname or location,
            "message": f"Added {location} to favorites",
            "favorites_count": len(self.favorite_locations)
        }
    
    @tool(description="Get recent weather query history")
    async def get_query_history(self, limit: int = 10) -> Dict[str, Any]:
        """Get the recent weather query history."""
        recent_queries = self.recent_queries[-limit:] if self.recent_queries else []
        
        return {
            "history": recent_queries,
            "total_queries": self.query_count,
            "limit": limit,
            "last_location": self.last_location,
            "message": f"Retrieved last {len(recent_queries)} weather queries from history"
        }
    
    @tool(description="Get favorite locations list")
    async def get_favorite_locations(self) -> Dict[str, Any]:
        """Get the list of favorite locations."""
        return {
            "favorites": self.favorite_locations,
            "count": len(self.favorite_locations),
            "message": f"You have {len(self.favorite_locations)} favorite locations"
        }
    
    @tool(description="Clear weather query history and reset data")
    async def clear_history(self) -> Dict[str, Any]:
        """Clear weather query history and reset state."""
        old_count = self.query_count
        
        # Clear query data but preserve favorites
        favorites_backup = self.favorite_locations.copy()
        self.recent_queries = []
        self.query_count = 0
        self.favorite_locations = favorites_backup
        self.last_location = None
        
        print(f"\n🧹 [Weather] Query history cleared ({old_count} queries removed)\n")
        
        return {
            "message": f"Weather query history cleared ({old_count} queries removed)",
            "favorites_preserved": len(self.favorite_locations),
            "cleared": True
        }