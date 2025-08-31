#!/usr/bin/env python3
# weather/weather.py - Weather data fetching using wttr.in API

import json
import re
from typing import Dict, Any, Optional
import httpx


class WeatherAPI:
    """Weather API client for wttr.in service."""
    
    @staticmethod
    def _clean_location(location: str) -> str:
        """Clean and normalize location string."""
        # Remove special characters and normalize
        location = re.sub(r'[^\w\s,.-]', '', location)
        location = location.strip()
        
        # If empty or only whitespace, use current location
        if not location:
            return ""
        
        # Replace spaces with + for URL encoding
        return location.replace(' ', '+')
    
    @staticmethod
    async def get_current_weather(client: httpx.AsyncClient, location: str = "") -> Dict[str, Any]:
        """Get current weather for a location."""
        clean_location = WeatherAPI._clean_location(location)
        
        try:
            # Get JSON format weather data
            response = await client.get(f"/{clean_location}?format=j1")
            response.raise_for_status()  # Raise exception for 4xx/5xx responses
            
            data = response.json()
            
            # Extract current conditions
            current = data.get('current_condition', [{}])[0]
            location_info = data.get('nearest_area', [{}])[0]
            
            return {
                "location": {
                    "name": location_info.get('areaName', [{'value': location or 'Current Location'}])[0]['value'],
                    "region": location_info.get('region', [{'value': ''}])[0]['value'],
                    "country": location_info.get('country', [{'value': ''}])[0]['value'],
                    "query": location or "auto-detected"
                },
                "current": {
                    "temperature_c": int(current.get('temp_C', 0)),
                    "temperature_f": int(current.get('temp_F', 0)),
                    "feels_like_c": int(current.get('FeelsLikeC', 0)),
                    "feels_like_f": int(current.get('FeelsLikeF', 0)),
                    "condition": current.get('weatherDesc', [{'value': 'Unknown'}])[0]['value'],
                    "humidity": int(current.get('humidity', 0)),
                    "wind_speed_kmh": int(current.get('windspeedKmph', 0)),
                    "wind_speed_mph": int(current.get('windspeedMiles', 0)),
                    "wind_direction": current.get('winddir16Point', 'N'),
                    "pressure_mb": int(current.get('pressure', 0)),
                    "visibility_km": int(current.get('visibility', 0)),
                    "uv_index": int(current.get('uvIndex', 0)),
                    "cloud_cover": int(current.get('cloudcover', 0))
                },
                "success": True,
                "timestamp": data.get('current_condition', [{}])[0].get('observation_time', 'unknown')
            }
            
        except httpx.TimeoutException:
            return {
                "error": "Weather service request timed out",
                "location": location,
                "success": False
            }
        except httpx.HTTPStatusError as e:
            return {
                "error": f"Weather service returned status {e.response.status_code}",
                "location": location,
                "success": False
            }
        except httpx.RequestError as e:
            return {
                "error": f"Network error: {str(e)}",
                "location": location,
                "success": False
            }
        except json.JSONDecodeError:
            return {
                "error": "Invalid response from weather service",
                "location": location,
                "success": False
            }
        except Exception as e:
            return {
                "error": f"Unexpected error: {str(e)}",
                "location": location,
                "success": False
            }
    
    @staticmethod
    async def get_forecast(client: httpx.AsyncClient, location: str = "", days: int = 3) -> Dict[str, Any]:
        """Get weather forecast for a location."""
        clean_location = WeatherAPI._clean_location(location)
        days = max(1, min(days, 3))  # Limit to 1-3 days
        
        try:
            response = await client.get(f"/{clean_location}?format=j1")
            response.raise_for_status()
            
            data = response.json()
            location_info = data.get('nearest_area', [{}])[0]
            
            # Extract forecast data
            forecast_days = []
            for day_data in data.get('weather', [])[:days]:
                day_forecast = {
                    "date": day_data.get('date', ''),
                    "max_temp_c": int(day_data.get('maxtempC', 0)),
                    "max_temp_f": int(day_data.get('maxtempF', 0)),
                    "min_temp_c": int(day_data.get('mintempC', 0)),
                    "min_temp_f": int(day_data.get('mintempF', 0)),
                    "condition": day_data.get('hourly', [{}])[0].get('weatherDesc', [{'value': 'Unknown'}])[0]['value'],
                    "sunrise": day_data.get('astronomy', [{}])[0].get('sunrise', ''),
                    "sunset": day_data.get('astronomy', [{}])[0].get('sunset', ''),
                    "moon_phase": day_data.get('astronomy', [{}])[0].get('moon_phase', ''),
                    "hourly_forecast": []
                }
                
                # Add hourly forecast (every 3 hours)
                for hour_data in day_data.get('hourly', []):
                    time = hour_data.get('time', '0').zfill(4)
                    formatted_time = f"{time[:2]}:{time[2:]}"
                    
                    hourly = {
                        "time": formatted_time,
                        "temp_c": int(hour_data.get('tempC', 0)),
                        "temp_f": int(hour_data.get('tempF', 0)),
                        "condition": hour_data.get('weatherDesc', [{'value': 'Unknown'}])[0]['value'],
                        "wind_speed_kmh": int(hour_data.get('windspeedKmph', 0)),
                        "humidity": int(hour_data.get('humidity', 0)),
                        "chance_of_rain": int(hour_data.get('chanceofrain', 0))
                    }
                    day_forecast["hourly_forecast"].append(hourly)
                
                forecast_days.append(day_forecast)
            
            return {
                "location": {
                    "name": location_info.get('areaName', [{'value': location or 'Current Location'}])[0]['value'],
                    "region": location_info.get('region', [{'value': ''}])[0]['value'],
                    "country": location_info.get('country', [{'value': ''}])[0]['value'],
                    "query": location or "auto-detected"
                },
                "forecast": forecast_days,
                "days_requested": days,
                "success": True
            }
            
        except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.RequestError, json.JSONDecodeError, Exception) as e:
            return {
                "error": f"Error getting forecast: {str(e)}",
                "location": location,
                "success": False
            }
    
    @staticmethod
    async def get_weather_summary(client: httpx.AsyncClient, location: str = "") -> Dict[str, Any]:
        """Get a simple text summary of current weather."""
        clean_location = WeatherAPI._clean_location(location)
        
        try:
            # Get simple text format
            response = await client.get(f"/{clean_location}?format=%l:+%c+%t+%f+%h+%w")
            response.raise_for_status()
            
            summary = response.text.strip()
            
            return {
                "location": location or "Current Location",
                "summary": summary,
                "success": True
            }
            
        except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.RequestError, Exception) as e:
            return {
                "error": f"Error getting weather summary: {str(e)}",
                "location": location,
                "success": False
            }