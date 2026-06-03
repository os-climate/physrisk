from physrisk.data.inventory import EmbeddedInventory


def test_get_indicator_id_display_names():
    inventory = EmbeddedInventory()
    indicator_ids = set(r.indicator_id for r in inventory.resources.values())
    static_mappings = {
        "days/above/35C": "Days per year with maximum temperature above 35C",
        "days_wbgt_above": "Days with wet bulb globe temperature above threshold",
        "flood_depth/nosub/5": "Flood depth, excluding subsidence, 5th percentile sea level rise",
        "flood_depth/nosub/95": "Flood depth, excluding subsidence, 95th percentile sea level rise",
        "flood_sop": "Flood standard of protection",
        "mean_work_loss/low": "Mean work loss for low intensity work",
        "mean_work_loss/medium": "Mean work loss for medium intensity work",
        "mean_work_loss/high": "Mean work loss for high intensity work",
        "months/spei3m/below/-2": "Months with 3 month SPEI below -2",
        "months/spei12m/below/index": "Months with 12 month SPEI below threshold",  # legacy
        "months/spei12m/below/threshold": "Months with 12 month SPEI below threshold",
        "max/daily/water_equivalent": "Max daily water-equivalent precipitation",
        "mean_degree_days/above/32C": "Mean degree days above 32°C",
        "mean_degree_days/below/index": "Mean degree days below threshold",
        "weeks_water_temp_above": "Weeks with water temperature above threshold",
        "wind_speed/3s": "3 second gust max wind speed",
    }
    mapping = {}
    import re

    for indicator_id in indicator_ids:
        match = re.match(r"days_tas/above/(\d+)", indicator_id)
        if match:
            mapping[indicator_id] = (
                rf"Days with average temperature above {match.group(1)}°C"
            )
        elif indicator_id in static_mappings:
            mapping[indicator_id] = static_mappings[indicator_id]
        else:
            mapping[indicator_id] = prettify(indicator_id)
    print(indicator_ids)


def prettify(indicator_id: str):
    return indicator_id.replace("_", " ").replace("/", " ").capitalize()
