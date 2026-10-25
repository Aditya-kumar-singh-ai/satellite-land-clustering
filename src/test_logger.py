
from experiment_logger import save_result

save_result(
    experiment="Logger Test",
    features=6,
    clusters=10,
    silhouette=0.3310,
    ari=0.1861
)

print("Logger test completed!")

