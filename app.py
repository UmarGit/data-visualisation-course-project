import io
import warnings
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import streamlit as st

warnings.filterwarnings("ignore")


@st.cache_data
def preprocess_temperature_data(data, cities):
    """
    Preprocesses temperature data by filling missing values, correcting years,
    and aggregating summary statistics.

    Parameters:
        data (DataFrame): The raw temperature dataset.
        cities (DataFrame): The dataset mapping cities to states.

    Returns:
        DataFrame: The cleaned temperature dataset.
    """
    cleaned_data = data.copy()

    # Fill missing states using a city-to-state mapping
    city_to_state = cities.set_index("name")["state"].to_dict()
    cleaned_data["State"] = cleaned_data.apply(
        lambda row: city_to_state.get(row["City"], row["State"]), axis=1
    )
    cleaned_data["State"].fillna("Unknown", inplace=True)

    # Correct day and year anomalies
    cleaned_data["Day"] = cleaned_data["Day"].replace(0, 1)
    cleaned_data["Year"] = cleaned_data["Year"].apply(
        lambda year: year if 1900 <= year <= 2024 else None
    )
    cleaned_data["Year"].fillna(method="ffill", inplace=True)
    cleaned_data["Year"].fillna(method="bfill", inplace=True)

    # Convert temperatures to Celsius
    cleaned_data["AvgTemperature"] = (cleaned_data["AvgTemperature"] - 32) * 5 / 9

    return cleaned_data


def plot_bar_chart(
    data,
    title,
    xlabel,
    ylabel,
    face_color="ffffff",
    figsize=(12, 6),
):
    """
    Plots a bar chart for a given data column.

    Parameters:
        data (Series): Data to plot.
        column (str): The column to visualize.
        title (str): Plot title.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        figsize (tuple): Dimensions of the figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    selection = st.radio(
        "Min/Max",
        ["Min", "Max"],
        captions=[
            "Minimum temperature of the region",
            "Maximum temperature of the region",
        ],
        key="Min",
        label_visibility="hidden",
    )

    if selection == "Min":
        temp = regional_avg_temp.min()
        region = regional_avg_temp.idxmin()
    else:
        temp = regional_avg_temp.max()
        region = regional_avg_temp.idxmax()

    ann_label = f"Max Temp: {temp:.2f}°C"
    ann_xy = (regional_avg_temp.index.get_loc(region), temp)

    ax.annotate(
        ann_label,
        xy=ann_xy,
        xytext=(0, 20),
        textcoords="offset points",
        ha="center",
        arrowprops=dict(facecolor="red", arrowstyle="->"),
        fontsize=10,
        color="red",
    )

    fig.set_facecolor(f"#{face_color}")
    ax.set_facecolor(f"#{face_color}")

    data.plot(kind="bar", ax=ax, color="#C7E171", edgecolor="black")
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xticklabels(
        [
            "Europe",
            "Africa",
            "North America",
            "Australia",
            "South America",
            "Asia",
            "Middle East",
        ]
    )
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    st.pyplot(fig)

    img = io.BytesIO()
    plt.savefig(img, format="png")

    st.download_button(
        label="Download Figure",
        data=img,
        file_name="avg_temp_by_region.png",
        mime="image/png",
    )


def plot_seasonal_trends(data, regions, face_color="ffffff", figsize=(12, 6)):
    """
    Plots seasonal temperature trends by region.

    Parameters:
        data (DataFrame): The dataset containing temperature data.
        regions (list): List of regions to include in the plot.
        figsize (tuple): Dimensions of the figure.
    """
    seasonal_data = (
        data.groupby(["Region", "Month"])["AvgTemperature"].mean().reset_index()
    )

    fig, ax = plt.subplots(figsize=figsize)

    fig.set_facecolor(f"#{face_color}")
    ax.set_facecolor(f"#{face_color}")

    for index, region in enumerate(regions):
        region_data = seasonal_data[seasonal_data["Region"] == region]

        if region_data.empty:
            print(f"Warning: No data available for region '{region}'. Skipping.")
            continue

        ax.plot(
            region_data["Month"],
            region_data["AvgTemperature"],
            label=region,
            color=mpl.colormaps["seasonal_trends"](index),
        )

        max_temp_row = region_data.loc[region_data["AvgTemperature"].idxmax()]

        ax.annotate(
            f'Highest: {max_temp_row["AvgTemperature"]:.1f}°C',
            xy=(max_temp_row["Month"], max_temp_row["AvgTemperature"]),
            xytext=(-10, -30),
            textcoords="offset points",
            ha="left",
            arrowprops=dict(
                facecolor=mpl.colormaps["seasonal_trends"](index), arrowstyle="->"
            ),
            fontsize=8,
            color=mpl.colormaps["seasonal_trends"](index),
        )

    ax.set_title("Seasonal Temperature Trends", fontsize=16)
    ax.set_xlabel("Month", fontsize=14)
    ax.set_ylabel("Average Temperature (°C)", fontsize=14)
    ax.legend(title="Region", fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_xticks(
        range(1, 13),
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ],
    )
    plt.set_cmap("seasonal_trends")
    st.pyplot(fig)

    img = io.BytesIO()
    plt.savefig(img, format="png")

    st.download_button(
        label="Download Figure",
        data=img,
        file_name="seasonal_trends_by_region.png",
        mime="image/png",
    )


def plot_temperature_trends(data, cities, face_color="ffffff", figsize=(12, 6)):
    """
    Plots temperature trends for selected cities in a specific year.

    Parameters:
        data (DataFrame): The dataset containing temperature data.
        cities (list): List of cities to include in the plot.
        face_color (str): Background color of the plot (hex).
        figsize (tuple): Dimensions of the figure.
    """
    selected_data = data[data["City"].isin(cities)]
    selected_data["Date"] = pd.to_datetime(selected_data[["Year", "Month", "Day"]])
    selected_year = st.selectbox(
        "Select Year To Visualise Trend",
        options=selected_data["Year"].unique().astype(int),
    )
    selected_data = selected_data[selected_data["Year"] == selected_year]

    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor(f"#{face_color}")
    ax.set_facecolor(f"#{face_color}")

    for index, city in enumerate(cities):
        city_data = selected_data[selected_data["City"] == city]
        ax.plot(
            city_data["Date"],
            city_data["AvgTemperature"],
            label=city,
            color=mpl.colormaps["temperature_trends"](index),
        )

        max_temp_point = city_data.loc[city_data["AvgTemperature"].idxmax()]

        ax.annotate(
            f'{city}: {max_temp_point["AvgTemperature"]:.1f}°C',
            xy=(max_temp_point["Date"], max_temp_point["AvgTemperature"]),
            xytext=(20, -60),
            textcoords="offset points",
            ha="center",
            arrowprops=dict(
                facecolor=mpl.colormaps["temperature_trends"](index), arrowstyle="->"
            ),
            fontsize=10,
            color=mpl.colormaps["temperature_trends"](index),
        )

    ax.set_title(
        f"Temperature Trends for Selected Cities ({selected_year})", fontsize=16
    )
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Average Temperature (°C)", fontsize=14)
    ax.legend(title="City", fontsize=12)
    ax.grid(alpha=0.3)
    plt.set_cmap("temperature_trends")
    st.pyplot(fig)

    img = io.BytesIO()
    plt.savefig(img, format="png")

    st.download_button(
        label="Download Figure",
        data=img,
        file_name="temp_trends_by_city.png",
        mime="image/png",
    )


def plot_global_heatmap(data, face_color="ffffff", figsize=(20, 10)):
    """
    Plots a global heatmap of average temperatures.

    Parameters:
        data (DataFrame): The dataset containing temperature data.
        figsize (tuple): Dimensions of the figure.
    """
    world = gpd.read_file(
        "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    )
    avg_temps = data.groupby("Country")["AvgTemperature"].mean().reset_index()
    world = world.merge(avg_temps, left_on="ADMIN", right_on="Country", how="left")

    fig, ax = plt.subplots(figsize=figsize)

    fig.set_facecolor(f"#{face_color}")
    ax.set_facecolor(f"#{face_color}")

    world.plot(
        column="AvgTemperature",
        cmap="global_heatmap",
        linewidth=0.5,
        ax=ax,
        edgecolor="black",
        legend=True,
        missing_kwds={"color": "white", "label": "No Data"},
    )

    highest_temp_country = avg_temps.loc[avg_temps["AvgTemperature"].idxmax()]
    ax.annotate(
        f'{highest_temp_country["Country"]}\n{highest_temp_country["AvgTemperature"]:.1f}°C',
        xy=(53.9485752, 24.3540069),
        xytext=(80, -100),
        textcoords="offset points",
        ha="center",
        arrowprops=dict(facecolor="red", arrowstyle="->"),
        fontsize=10,
        color="red",
    )

    ax.set_title("Global Temperature Heatmap", fontsize=20)
    ax.set_xticks([])
    ax.set_yticks([])
    st.pyplot(fig)

    img = io.BytesIO()
    plt.savefig(img, format="png")

    st.download_button(
        label="Download Figure",
        data=img,
        file_name="global_temperature_heatmap.png",
        mime="image/png",
    )


def generate_custom_color_maps():
    """
    Generates and registers custom colormaps for different visualizations.
    """
    seasonal_trends_cmap = ListedColormap(
        ["#C77CFF", "#FEA903", "#C7E171", "#FFDAB9", "#CCE2CE", "#FFF200", "#DCD0FF"],
        name="seasonal_trends",
    )
    temperature_trends_cmap = ListedColormap(
        ["#C7E171", "#C77CFF", "#FEA903", "#E0EAFF"], name="temperature_trends"
    )
    global_heatmap_colors = [f"#FEA9{i:02X}" for i in range(16, 256, 24)]
    global_heatmap_cmap = ListedColormap(global_heatmap_colors, name="global_heatmap")

    for cmap_name in ["seasonal_trends", "temperature_trends", "global_heatmap"]:
        if cmap_name in mpl.colormaps:
            mpl.colormaps.unregister(name=cmap_name)

    mpl.colormaps.register(cmap=seasonal_trends_cmap)
    mpl.colormaps.register(cmap=temperature_trends_cmap)
    mpl.colormaps.register(cmap=global_heatmap_cmap)


markdown = """
# **A World in `Flux`**
### *Unraveling Global Temperature Trends*

---

We will explore **global temperature patterns**, highlighting **regional disparities** and the **impact of climate change**.
Through interactive **visualisations**, we will delve into the **past, present, and future** of our planet's climate.

---

##### **Data Visualisation: Course Project**
- **Date:** 6th December 2024
- **Professor:** Грандилевский Алексей Ильич
- **Author:** *Умар Ахмед*

---
"""

st.set_page_config(
    page_title="Data Visualisation: Course Project",
)

st.markdown(markdown)

# File upload
uploaded_temp_file = st.file_uploader("Upload City Temperature CSV", type=["csv"])

if uploaded_temp_file:
    temp_data = pd.read_csv(uploaded_temp_file)
    city_data = pd.read_csv(
        "https://docs.google.com/spreadsheets/d/"
        + "1o47G_i5MTXnn5EFmhJIRuOLuxLCpUXwTFZDXO1V9YHE"
        + "/export?format=csv",
    )

    cleaned_data = preprocess_temperature_data(temp_data, city_data)
    generate_custom_color_maps()

    # Regional average temperatures
    st.subheader("Average Temperature by Region")
    regional_avg_temp = (
        cleaned_data.groupby("Region")["AvgTemperature"].mean().sort_values()
    )
    plot_bar_chart(
        regional_avg_temp,
        title="Average Temperature by Region",
        xlabel="Region",
        ylabel="Average Temperature (°C)",
        face_color="FFFFFF",
    )

    # Seasonal trends
    st.subheader("Seasonal Temperature Trends")
    selected_regions = st.multiselect(
        "Select Regions to Display",
        options=cleaned_data["Region"].unique(),
        default=cleaned_data["Region"].unique()[:3],
    )
    plot_seasonal_trends(
        cleaned_data,
        selected_regions,
        face_color="FFFFFF",
    )

    # Temperature trends for cities
    st.subheader("Temperature Trends for Selected Cities")
    selected_cities = st.multiselect(
        "Select Cities to Display",
        options=cleaned_data["City"].unique(),
        default=["Los Angeles", "Moscow", "Karachi"],
    )
    plot_temperature_trends(
        cleaned_data,
        selected_cities,
        face_color="FFFFFF",
    )

    # Global heatmap
    st.subheader("Global Temperature Heatmap")
    plot_global_heatmap(
        cleaned_data,
        face_color="FFFFFF",
    )
else:
    st.warning("Please upload the City Temperature file.")
