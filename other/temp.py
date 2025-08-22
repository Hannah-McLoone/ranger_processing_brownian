import rasterio
import matplotlib.pyplot as plt

# Open the TIF file
import rasterio

with rasterio.open("ndvi.tif") as src:
    image = src.read(1)  # Read the first band

    # Determine rows corresponding to 20% and 70% of the height


# Display as heatmap
plt.imshow(image, cmap='hot')
plt.colorbar(label='slope')
plt.title("TIF Heatmap (Rasterio)")
plt.show()
