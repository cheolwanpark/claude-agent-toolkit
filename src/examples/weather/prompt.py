#!/usr/bin/env python3
# weather/prompt.py - System prompts and templates for weather demo

# System prompt for the weather agent
WEATHER_SYSTEM_PROMPT = """You are a professional weather assistant with access to real-time weather data from around the world.

Your capabilities include:
- Current weather conditions for any location
- Multi-day weather forecasts (up to 3 days)
- Weather comparisons between locations
- Weather history tracking and favorites management
- Detailed meteorological analysis and insights

Guidelines:
1. Always use the weather tools to get real-time data - never provide outdated or guessed information
2. Provide comprehensive weather information including temperature, conditions, humidity, wind, etc.
3. Interpret weather data in a user-friendly way with practical advice
4. Offer context about what the weather means for daily activities
5. Be proactive about severe weather warnings or notable conditions
6. Use location nicknames and favorites when available
7. Provide temperature in both Celsius and Fahrenheit when relevant

When users ask about weather:
1. Clarify location if ambiguous
2. Use the appropriate weather tool for their specific need
3. Interpret the data with practical insights
4. Suggest related information that might be helpful
5. Offer to save frequently requested locations as favorites

Weather Interpretation Tips:
- Mention "feels like" temperature when significantly different from actual
- Highlight high UV index, strong winds, or low visibility when present
- Suggest appropriate clothing or activities based on conditions
- Warn about precipitation probability and timing
- Comment on air pressure trends if relevant

Remember: You have access to query history and can reference previous weather requests."""

# Template for weather requests
WEATHER_REQUEST_TEMPLATE = """I need to get weather information for: {location}

Let me fetch the current real-time weather data for you."""

# Template for forecast requests
FORECAST_REQUEST_TEMPLATE = """I'll get a {days}-day weather forecast for {location}.

This will help you plan ahead for the coming days."""

# Template for weather comparisons
COMPARISON_TEMPLATE = """I'll compare the current weather between {location1} and {location2}.

This will help you understand the weather differences between these locations."""

# Template for weather analysis
ANALYSIS_TEMPLATE = """Based on the current weather data for {location}:

Weather Conditions: {condition}
Temperature: {temperature}°C ({temp_f}°F) - Feels like {feels_like}°C ({feels_like_f}°F)
Humidity: {humidity}%
Wind: {wind_speed} km/h {wind_direction}

Analysis: {analysis}

Recommendations: {recommendations}"""

# Welcome message for the weather demo
WELCOME_MESSAGE = """🌤️ Welcome to the Claude Agent Toolkit Weather Demo!

I'm your weather assistant powered by the Claude Agent Toolkit framework with real-time weather data from wttr.in. I can help you with:

☀️ **Current Conditions**: Get real-time weather for any location worldwide
🌦️ **Forecasts**: 1-3 day forecasts with hourly details
🔄 **Comparisons**: Compare weather between multiple locations
⭐ **Favorites**: Save frequently checked locations
📊 **History**: Track your weather queries and patterns
🌍 **Global Coverage**: Weather data for cities, countries, landmarks, and coordinates

**Example queries:**
- "What's the weather like in Tokyo right now?"
- "Give me a 3-day forecast for Paris, France"
- "Compare the weather in New York and Los Angeles"
- "What should I wear in London today?"
- "Is it going to rain in Seattle this weekend?"
- "Add San Francisco to my favorites"

I provide comprehensive weather data including temperature, humidity, wind, visibility, UV index, and practical advice for your daily activities.

What weather information can I help you with today?"""

# Template for severe weather warnings
SEVERE_WEATHER_TEMPLATE = """⚠️ **Weather Advisory for {location}**

{warning_type}: {description}

**Current Conditions:**
- Temperature: {temperature}°C ({temp_f}°F)
- Conditions: {condition}
- {additional_details}

**Recommendations:**
{safety_advice}

Stay safe and monitor local weather services for updates."""

# Template for travel weather advice
TRAVEL_ADVICE_TEMPLATE = """✈️ **Travel Weather Advisory**

**Departure**: {departure_location}
{departure_weather}

**Destination**: {destination_location}
{destination_weather}

**Travel Recommendations:**
{packing_advice}
{timing_advice}
{general_advice}"""

# Template for activity recommendations
ACTIVITY_RECOMMENDATIONS = {
    "outdoor_sports": [
        "Great weather for outdoor sports and activities!",
        "Perfect conditions for running, cycling, or hiking",
        "Ideal weather for outdoor events and gatherings"
    ],
    "indoor_activities": [
        "Better to plan indoor activities today",
        "Perfect weather for museums, shopping, or cozy indoor time",
        "Great day for indoor sports or entertainment"
    ],
    "mixed": [
        "Some outdoor time is possible with proper preparation",
        "Plan flexible activities that can move indoors if needed",
        "Check conditions again before heading out"
    ]
}

# Error message templates for weather
ERROR_TEMPLATES = {
    "location_not_found": "⚠️ I couldn't find weather data for '{location}'. Please check the spelling or try a more specific location name.",
    "service_unavailable": "⚠️ The weather service is temporarily unavailable. Please try again in a few moments.",
    "network_error": "⚠️ Unable to connect to weather service. Please check your internet connection.",
    "invalid_request": "⚠️ Please provide a valid location name or coordinates for weather information.",
    "rate_limited": "⚠️ Too many weather requests. Please wait a moment before requesting more weather data."
}

# Success message templates for weather
SUCCESS_TEMPLATES = {
    "current_weather": "🌤️ Current weather conditions retrieved successfully!",
    "forecast_retrieved": "📅 Weather forecast retrieved successfully!",
    "comparison_complete": "🔄 Weather comparison completed!",
    "location_added": "⭐ Location added to your favorites!",
    "history_retrieved": "📊 Weather query history retrieved!",
    "data_updated": "🔄 Weather data has been updated!"
}

# Weather condition interpretations
CONDITION_INTERPRETATIONS = {
    "clear": "Perfect clear skies with excellent visibility",
    "sunny": "Bright and sunny conditions ideal for outdoor activities",
    "partly_cloudy": "Mix of sun and clouds with generally pleasant conditions",
    "cloudy": "Overcast skies but typically dry conditions",
    "light_rain": "Light precipitation - an umbrella would be helpful",
    "rain": "Active rainfall - waterproof clothing recommended",
    "heavy_rain": "Significant rainfall - indoor activities preferred",
    "snow": "Snowy conditions - dress warmly and drive carefully",
    "thunderstorm": "Stormy weather - stay indoors when possible",
    "fog": "Reduced visibility - take extra care when traveling",
    "windy": "Strong winds - secure loose objects and dress accordingly"
}

# Temperature comfort guidelines
TEMPERATURE_COMFORT = {
    "very_cold": "Very cold conditions - heavy winter clothing essential",
    "cold": "Cold weather - warm layers and winter accessories needed",
    "cool": "Cool conditions - light jacket or sweater recommended",
    "mild": "Comfortable mild temperatures - light layers work well",
    "warm": "Pleasant warm weather - perfect for most outdoor activities",
    "hot": "Hot conditions - stay hydrated and seek shade when possible",
    "very_hot": "Very hot weather - limit outdoor exposure and drink plenty of fluids"
}