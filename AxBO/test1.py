from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.registry import Models

# Initialize AxClient
ax_client = AxClient()

# Define search space
ax_client.create_experiment(
    name="car_preference_experiment",
    parameters=[
        {"name": "color", "type": "choice", "values": ["red", "blue", "green"]},
        {"name": "fuel_type", "type": "choice", "values": ["gasoline", "diesel", "electric"]},
        {"name": "brand", "type": "choice", "values": ["BrandA", "BrandB", "BrandC"]}
    ],
    objectives={"user_preference": ObjectiveProperties(minimize=False)}
)


# Function to get user choice
def get_user_choice(option1, option2):
    print(f"Option 1: Color: {option1['color']}, Fuel Type: {option1['fuel_type']}, Brand: {option1['brand']}")
    print(f"Option 2: Color: {option2['color']}, Fuel Type: {option2['fuel_type']}, Brand: {option2['brand']}")
    choice = input("Which car do you prefer? (Enter 1 or 2): ")
    return 1 if choice == '1' else 2


# Run trials with GPEI optimization
for i in range(10):
    # Use GPEI model to get the next most informative options
    gpei_model = Models.GPEI(experiment=ax_client.experiment)
    best_arm, trial_index = ax_client.get_next_trial(model=gpei_model)

    # Generate second option
    second_arm, second_trial_index = ax_client.get_next_trial(model=gpei_model)

    # Present the options to the user
    user_choice = get_user_choice(best_arm, second_arm)

    # Record the result
    if user_choice == 1:
        ax_client.complete_trial(trial_index=trial_index, raw_data=1.0)
        ax_client.complete_trial(trial_index=second_trial_index, raw_data=0.0)
    else:
        ax_client.complete_trial(trial_index=trial_index, raw_data=0.0)
        ax_client.complete_trial(trial_index=second_trial_index, raw_data=1.0)

# Display optimization trace
print(ax_client.get_optimization_trace())
