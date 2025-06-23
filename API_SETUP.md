# 🔑 API Setup Guide for BioFlux Real Data Integration

## Quick Start

1. **Copy the environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Edit the `.env` file** with your API keys (see instructions below)

3. **Test your configuration:**
   ```bash
   python config.py
   ```

4. **Run the interactive setup wizard (optional):**
   ```bash
   python config.py setup
   ```

## 🆓 Free API Keys You Can Get

### 1. OpenWeatherMap (Weather Data)
- **What it provides:** Current weather, forecasts, historical data
- **Free tier:** 1,000 API calls per day
- **Sign up:** https://openweathermap.org/api
- **Setup:**
  1. Create account
  2. Go to API Keys section
  3. Copy your API key
  4. Add to `.env`: `OPENWEATHER_API_KEY=your_key_here`

### 2. Sentinel Hub (Satellite NDVI Data)
- **What it provides:** Sentinel-2 satellite imagery, NDVI calculation
- **Free tier:** 1,000 requests per month
- **Sign up:** https://www.sentinel-hub.com/
- **Setup:**
  1. Create account
  2. Create a new configuration
  3. Note your Instance ID
  4. Add to `.env`: `SENTINELHUB_INSTANCE_ID=your_instance_id`

### 3. NASA EarthData (Additional Satellite Data)
- **What it provides:** MODIS, Landsat, and other NASA satellite products
- **Free tier:** No strict limits (rate limited)
- **Sign up:** https://urs.earthdata.nasa.gov/
- **Setup:**
  1. Create account
  2. Add credentials to `.env`:
     ```
     NASA_EARTHDATA_USERNAME=your_username
     NASA_EARTHDATA_PASSWORD=your_password
     ```

### 4. USGS Elevation (Free - No API Key Required)
- **What it provides:** Digital elevation model data for the US and global coverage
- **Free tier:** Unlimited (rate limited)
- **Sign up:** Not required
- **API:** https://nationalmap.gov/epqs/
- **Setup:** No setup needed - works automatically

## 📝 Example .env File

```bash
# Weather APIs
OPENWEATHER_API_KEY=abc123def456ghi789

# Satellite Data APIs  
SENTINELHUB_INSTANCE_ID=12345678-1234-1234-1234-123456789abc
NASA_EARTHDATA_USERNAME=your_nasa_username
NASA_EARTHDATA_PASSWORD=your_nasa_password

# Settings
DEBUG_MODE=false
DEMO_MODE_FALLBACK=true
DATA_CACHE_DIR=./data_cache
```

## 🔧 Configuration Commands

```bash
# Check current API configuration status
python config.py

# Run interactive setup wizard
python config.py setup

# Test real data integration with your APIs
python real_data_api.py

# Run demo with location-specific data
python quick_real_data_demo.py
```

## 🎯 What Each API Provides

| API | Data Type | Update Frequency | Usage in BioFlux |
|-----|-----------|------------------|-------------------|
| OpenWeatherMap | Weather | Real-time | Temperature, humidity, precipitation affecting agent behavior |
| Sentinel Hub | NDVI/Vegetation | Every 5 days | Vegetation density affecting food availability |
| NASA EarthData | Satellite imagery | Daily/weekly | Historical trends, validation data |
| USGS Elevation | Elevation/Terrain | Static | Terrain effects on movement (free service) |

## 💭 Demo Mode (No API Keys Required)

If you don't set up API keys, BioFlux will automatically use **realistic synthetic data** that simulates:

- ✅ Seasonal weather patterns
- ✅ Location-specific climate (Arctic, tropical, desert, etc.)
- ✅ Realistic NDVI vegetation patterns
- ✅ Spatial correlation in environmental data

This lets you test and develop without any API setup!

## 🔒 Security Best Practices

1. **Never commit `.env` files** to version control
2. **Use different API keys** for development/production
3. **Rotate API keys** periodically
4. **Monitor API usage** to avoid hitting rate limits
5. **Use environment variables** in production deployments

## ⚠️ Rate Limits and Quotas

| Service | Free Limit | Reset Period | Overage |
|---------|------------|--------------|---------|
| OpenWeatherMap | 1,000 calls | Daily | Paid plans available |
| Sentinel Hub | 1,000 requests | Monthly | Paid plans available |
| NASA EarthData | Rate limited | N/A | Free |
| USGS Elevation | Rate limited | N/A | Free |

## 🚀 Getting Started

1. **Start with OpenWeatherMap** - easiest to set up, immediate results
2. **Add Sentinel Hub** - for satellite vegetation data
3. **Optionally add NASA EarthData** - for research-grade data
4. **Test with different locations** - see how real environmental data affects your simulations

## 🆘 Troubleshooting

**Problem: "Import requests could not be resolved"**
- Solution: `uv add requests` or ensure you're in the right virtual environment

**Problem: API returns 401/403 errors**
- Solution: Check your API key is correct and active

**Problem: "Module not found" errors**
- Solution: Run with `uv run python script.py` to use the project environment

**Problem: API rate limits exceeded**
- Solution: Enable `DEMO_MODE_FALLBACK=true` in `.env` file

---

**Ready to get started?** Copy `.env.example` to `.env` and fill in your first API key! 🚀
