import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define timeframe around original time that is searched
timeframe = 1

# Define paths
ball_files_directory = r"C:\Users\steffen.barthel\TSG 1899 Hoffenheim Fußball-Spielbetriebs GmbH\Tactical Learning (Helix) - General\Data\Germany WEUROs_Hawkeye"
folder_path = r"C:\Users\steffen.barthel\OneDrive - TSG 1899 Hoffenheim Fußball-Spielbetriebs GmbH\Dokumente\VS Code\.vscode\filter_sequences\Anniek\20240920_Files\sequence_L"

import os
import pandas as pd
import json

# Step 1: Load Sequences (no changes here, as it is already efficient)
def load_sequences(folder_path):
    sequences = []
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.startswith("sequence_") and file_name.endswith(".csv"):
                sequence_file = os.path.join(root, file_name)
                seq_df = pd.read_csv(sequence_file, delimiter=';')
                sequences.append((seq_df, sequence_file))
    return sequences

# Step 2: Extract relevant ball data based on each BallReceipt time individually
def extract_relevant_ball_data(ball_files_directory, sequence_df, timeframe=5.0):
    # Create a temporary column 'GameID_clean' by removing 'M' from GameID for processing
    sequence_df['GameID_clean'] = sequence_df['GameID'].astype(str).str.replace('M', '')

    # Pre-group sequence_df by GameID_clean and Half to reduce repetitive filtering
    sequence_df['Half'] = sequence_df['Half'].astype(int)
    sequence_df['BallReceipt'] = sequence_df['BallReceipt'].astype(float)

    # Group sequences by cleaned GameID and Half for faster lookup
    grouped_sequences = sequence_df.groupby(['GameID_clean', 'Half'])

    relevant_ball_data = []

    # Iterate through all the files in the directory
    for root, _, files in os.walk(ball_files_directory):
        for file in files:
            if file.endswith('.ball'):
                # Extract game_id, half, minute, and added_minute from the filename
                parts = file.replace('.football.samples.ball', '').split('_')
                game_id = parts[2]
                half = int(parts[3])
                minute = int(parts[4])
                added_minute = int(parts[5]) if len(parts) > 5 else 0

                # Calculate the base time for this file
                if half == 1:
                    base_time = (minute + added_minute - 1) * 60
                elif half == 2:
                    base_time = (minute + added_minute - 46) * 60
                elif half == 3:
                    base_time = (minute + added_minute - 91) * 60
                else:
                    base_time = (minute + added_minute - 106) * 60

                # Only proceed if there are sequences for this game_id and half
                if (game_id, half) in grouped_sequences.groups:
                    relevant_sequences = grouped_sequences.get_group((game_id, half))

                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if 'samples' in data and 'ball' in data['samples']:
                            for sample in data['samples']['ball']:
                                time = sample['time']
                                ball_speed = sample['speed']['mps']
                                ball_x = sample['pos'][0]
                                ball_y = sample['pos'][1]
                                ball_z = sample['pos'][2]

                                # Calculate the absolute time
                                abs_time = base_time + time

                                # Filter the relevant sequences based on the ball receipt time
                                time_start_range = relevant_sequences['BallReceipt'] - timeframe
                                time_end_range = relevant_sequences['BallReceipt'] + timeframe

                                match = (time_start_range <= abs_time) & (abs_time <= time_end_range)

                                if match.any():
                                    relevant_ball_data.append([game_id, half, abs_time, ball_speed, ball_x, ball_y, ball_z])

    # Create a DataFrame from the extracted ball data
    ball_df = pd.DataFrame(relevant_ball_data, columns=['game_id', 'half', 'time', 'ball_speed', 'ball_x', 'ball_y', 'ball_z'])
    ball_df.sort_values(by=['game_id', 'half', 'time'], inplace=True)
    ball_df.reset_index(drop=True, inplace=True)

    return ball_df

# Step 3: Process Ball Data (acceleration, direction change, scoring)
def process_ball_data(ball_data_df):
    ball_data_df['acceleration'] = ball_data_df.groupby(['game_id', 'half'])['ball_speed'].diff() / ball_data_df.groupby(['game_id', 'half'])['time'].diff()
    ball_data_df['acceleration'].fillna(0, inplace=True)
    
    # Calculate direction change
    ball_data_df['dx'] = ball_data_df.groupby(['game_id', 'half'])['ball_x'].diff()
    ball_data_df['dy'] = ball_data_df.groupby(['game_id', 'half'])['ball_y'].diff()
    ball_data_df['dz'] = ball_data_df.groupby(['game_id', 'half'])['ball_z'].diff()
    ball_data_df['magnitude'] = np.sqrt(ball_data_df['dx']**2 + ball_data_df['dy']**2 + ball_data_df['dz']**2)
    
    ball_data_df['prev_dx'] = ball_data_df['dx'].shift(-1)
    ball_data_df['prev_dy'] = ball_data_df['dy'].shift(-1)
    ball_data_df['prev_dz'] = ball_data_df['dz'].shift(-1)
    ball_data_df['prev_magnitude'] = ball_data_df['magnitude'].shift(-1)

    ball_data_df['dot_product'] = (ball_data_df['dx'] * ball_data_df['prev_dx'] +
                                   ball_data_df['dy'] * ball_data_df['prev_dy'] +
                                   ball_data_df['dz'] * ball_data_df['prev_dz'])

    # Ensure the input to arccos is in the range [-1, 1] to avoid invalid values
    cosine_values = ball_data_df['dot_product'] / (ball_data_df['magnitude'] * ball_data_df['prev_magnitude'])
    cosine_values = np.clip(cosine_values, -1, 1)  # Clip values to be between -1 and 1
    ball_data_df['angle_change'] = np.arccos(cosine_values)

    ball_data_df['angle_change'] = np.degrees(ball_data_df['angle_change'])
    ball_data_df['angle_change'].replace([np.inf, -np.inf], np.nan, inplace=True)
    ball_data_df['angle_change'].fillna(0, inplace=True)

    ball_data_df.drop(columns=['dx', 'dy', 'dz', 'magnitude', 'prev_dx', 'prev_dy', 'prev_dz', 'prev_magnitude', 'dot_product'], inplace=True)
    
    # Scoring
    def log_score(x, x_max, x_min=1):
        return np.log10(np.clip(x, x_min, x_max)) / np.log10(x_max) * 50

    accel_99th = ball_data_df[ball_data_df['acceleration'] > 0]['acceleration'].quantile(0.99)
    decel_99th = ball_data_df[ball_data_df['acceleration'] < 0]['acceleration'].quantile(0.01)

    ball_data_df['acceleration_score'] = -log_score(ball_data_df['acceleration'], accel_99th)
    ball_data_df.loc[ball_data_df['acceleration'] < 0, 'acceleration_score'] = log_score(-ball_data_df['acceleration'], -decel_99th)
    ball_data_df['acceleration_score'] = ball_data_df['acceleration_score'].clip(-50, 50)

    angle_99th = ball_data_df['angle_change'].quantile(0.99)
    ball_data_df['direction_change_score'] = log_score(ball_data_df['angle_change'], angle_99th)

    ball_data_df['total_score'] = ball_data_df['acceleration_score'] + ball_data_df['direction_change_score']
    
    return ball_data_df

# Step 4: Update Sequences with `time_news` and add extra columns
def update_sequences_with_time_news(ball_data_df, sequence_df):
    filtered_df = sequence_df.copy()
    filtered_df['BallReceipt_original'] = filtered_df['BallReceipt']

    ball_x_list = []
    ball_y_list = []
    ball_z_list = []
    ball_speed_list = []
    acceleration_list = []
    angle_change_list = []
    acceleration_score_list = []
    direction_change_score_list = []
    total_score_list = []

    time_news = []
    for _, row in filtered_df.iterrows():
        ball_receipt = row['BallReceipt']
        game_id = row['GameID_clean']
        half = row['Half']

        time_range = ball_data_df[(ball_data_df['game_id'] == game_id) & (ball_data_df['half'] == half) & 
                                  (ball_data_df['time'] >= ball_receipt - timeframe) & 
                                  (ball_data_df['time'] <= ball_receipt + timeframe)]

        if not time_range.empty:
            max_mean_row = time_range.loc[time_range['acceleration'].idxmin()]
            time_news.append(max_mean_row['time'])
            # Extract ball positions
            ball_x_list.append(max_mean_row['ball_x'])
            ball_y_list.append(max_mean_row['ball_y'])
            ball_z_list.append(max_mean_row['ball_z'])
            # Extract additional fields
            ball_speed_list.append(max_mean_row['ball_speed'])
            acceleration_list.append(max_mean_row['acceleration'])
            angle_change_list.append(max_mean_row['angle_change'])
            acceleration_score_list.append(max_mean_row['acceleration_score'])
            direction_change_score_list.append(max_mean_row['direction_change_score'])
            total_score_list.append(max_mean_row['total_score'])
        else:
            time_news.append(None)
            ball_x_list.append(None)
            ball_y_list.append(None)
            ball_z_list.append(None)
            ball_speed_list.append(None)
            acceleration_list.append(None)
            angle_change_list.append(None)
            acceleration_score_list.append(None)
            direction_change_score_list.append(None)
            total_score_list.append(None)

    filtered_df['BallReceipt'] = time_news
    filtered_df['BallReceipt_time_difference'] = filtered_df['BallReceipt_original'] - filtered_df['BallReceipt']
    filtered_df['ball_x'] = ball_x_list
    filtered_df['ball_y'] = ball_y_list
    filtered_df['ball_z'] = ball_z_list
    # Add the new variables to filtered_df
    filtered_df['ball_speed'] = ball_speed_list
    filtered_df['acceleration'] = acceleration_list
    filtered_df['angle_change'] = angle_change_list
    filtered_df['acceleration_score'] = acceleration_score_list
    filtered_df['direction_change_score'] = direction_change_score_list
    filtered_df['total_score'] = total_score_list

    return filtered_df

# Step 8a: Calculate z_ball at time_new
def calculate_z_ball(ball_df, filtered_df):
    z_values = []

    for _, row in filtered_df.iterrows():
        game_id = row['GameID_clean']
        half = row['Half']
        time_new = row['BallReceipt']  # time_new is already the updated BallReceipt

        # Get z_ball value at time_new
        z_value = ball_df[(ball_df['game_id'] == game_id) & 
                          (ball_df['half'] == half) & 
                          (ball_df['time'] == time_new)]['ball_z'].values
        if z_value.size > 0:
            z_values.append(z_value[0])
        else:
            z_values.append(np.nan)

    filtered_df['z_ball'] = z_values
    return filtered_df

# Step 8b: Calculate z_ball_max within timeframe before time_new
def calculate_z_ball_max(ball_df, filtered_df):
    z_max_values = []

    for _, row in filtered_df.iterrows():
        game_id = row['GameID_clean']
        half = row['Half']
        time_new = row['BallReceipt']  # time_new is already the updated BallReceipt

        # Calculate z_ball_max within timeframe before time_new
        past_data = ball_df[(ball_df['game_id'] == game_id) & 
                            (ball_df['half'] == half) & 
                            (ball_df['time'] >= time_new - timeframe) & 
                            (ball_df['time'] < time_new)]
        if not past_data.empty:
            z_max = past_data['ball_z'].max()
            z_max_values.append(z_max)
        else:
            z_max_values.append(np.nan)

    filtered_df['z_ball_max'] = z_max_values
    return filtered_df

# Step 8c: Calculate xy_ball_movement within timeframe after time_new
def calculate_xy_ball_movement(ball_df, filtered_df):
    xy_movements = []

    for _, row in filtered_df.iterrows():
        game_id = row['GameID_clean']
        half = row['Half']
        time_new = row['BallReceipt']  # time_new is already the updated BallReceipt

        # Calculate xy_ball_movement within timeframe after time_new
        future_data = ball_df[(ball_df['game_id'] == game_id) & 
                              (ball_df['half'] == half) & 
                              (ball_df['time'] >= time_new) & 
                              (ball_df['time'] <= time_new + timeframe)]
        if not future_data.empty:
            xy_movement = np.sqrt((future_data['ball_x'] - row['ball_x'])**2 + 
                                  (future_data['ball_y'] - row['ball_y'])**2).max()
            xy_movements.append(xy_movement)
        else:
            xy_movements.append(np.nan)

    filtered_df['xy_ball_movement'] = xy_movements
    return filtered_df

# Step 8d: Calculate 3m_reaction_time
def calculate_3m_reaction_time(ball_df, filtered_df):
    reaction_times = []

    for _, row in filtered_df.iterrows():
        game_id = row['GameID_clean']
        half = row['Half']
        time_new = row['BallReceipt']  # time_new is already the updated BallReceipt

        # Calculate 3m_reaction_time within timeframe before time_new
        past_data = ball_df[(ball_df['game_id'] == game_id) & 
                            (ball_df['half'] == half) & 
                            (ball_df['time'] >= time_new - timeframe) & 
                            (ball_df['time'] < time_new)]

        # Find the minimal timestamp when the ball was within 3 meters
        within_3m = past_data[np.sqrt((past_data['ball_x'] - row['ball_x'])**2 + 
                                      (past_data['ball_y'] - row['ball_y'])**2) <= 3]
        if not within_3m.empty:
            reaction_time = within_3m['time'].min()
            reaction_times.append(time_new - reaction_time)  # Calculate the reaction time
        else:
            reaction_times.append(np.nan)

    filtered_df['3m_reaction_time'] = reaction_times
    return filtered_df

import matplotlib.pyplot as plt

# Step 10: Plotting function for each BallReceipt
def plot_acceleration_and_angle_change(ball_data_df, filtered_df, sequence_file):
    # Create a figure with subplots for each BallReceipt
    num_receipts = len(filtered_df)
    fig, axs = plt.subplots(num_receipts, figsize=(10, num_receipts * 4))
    
    if num_receipts == 1:
        axs = [axs]  # Ensure axs is iterable even if there's only one BallReceipt

    for i, row in filtered_df.iterrows():
        game_id = row['GameID_clean']
        half = row['Half']
        ball_receipt_time = row['BallReceipt']  # New BallReceipt (time_new)
        ball_receipt_original = row['BallReceipt_original']  # Original BallReceipt
        scene_nr = row['SceneNr']

        # Define the timeframe around the BallReceipt_original as the baseline
        time_start = ball_receipt_original - timeframe
        time_end = ball_receipt_original + timeframe

        # Filter the ball data for this BallReceipt's timeframe
        time_range_data = ball_data_df[(ball_data_df['game_id'] == game_id) & 
                                       (ball_data_df['half'] == half) & 
                                       (ball_data_df['time'] >= time_start) & 
                                       (ball_data_df['time'] <= time_end)]

        # Plot acceleration on the left axis
        axs[i].plot(time_range_data['time'], time_range_data['acceleration'], label='Acceleration', color='b')
        axs[i].set_xlabel('Time (seconds)')
        axs[i].set_ylabel('Acceleration', color='b')
        axs[i].tick_params(axis='y', labelcolor='b')

        # Create a second y-axis for angle change
        ax2 = axs[i].twinx()
        ax2.plot(time_range_data['time'], time_range_data['angle_change'], label='Angle Change', color='g')
        ax2.set_ylabel('Angle Change (degrees)', color='g')
        ax2.tick_params(axis='y', labelcolor='g')

        # Add vertical line for the BallReceipt_original (the baseline)
        axs[i].axvline(x=ball_receipt_original, color='orange', linestyle='--', label='BallReceipt_Original')
        
        # Add vertical line for the BallReceipt (new_time)
        axs[i].axvline(x=ball_receipt_time, color='r', linestyle='--', label='BallReceipt')

        # Set the title for the subplot using SceneNr
        axs[i].set_title(f'SceneNr {scene_nr}: Acceleration and Angle Change')

        # Add legends
        axs[i].legend(loc='upper left')
        ax2.legend(loc='upper right')

    # Adjust the layout to avoid overlap
    plt.tight_layout()

    # Save the plot to a file, one image per sequence file
    output_plot_file = os.path.splitext(sequence_file)[0] + '_plots.png'
    plt.savefig(output_plot_file, dpi=300)
    plt.close()
    print(f"Saved plot file: {output_plot_file}")

# Step 10: Main function to run the steps with additional calculations
def main(folder_path, ball_files_directory):
    sequences = load_sequences(folder_path)

    for seq_df, sequence_file in sequences:
        ball_data_df = extract_relevant_ball_data(ball_files_directory, seq_df)
        ball_data_df = process_ball_data(ball_data_df)
        updated_seq_df = update_sequences_with_time_news(ball_data_df, seq_df)

        # Additional calculations based on the updated sequence
        updated_seq_df = calculate_z_ball(ball_data_df, updated_seq_df)
        updated_seq_df = calculate_z_ball_max(ball_data_df, updated_seq_df)
        updated_seq_df = calculate_xy_ball_movement(ball_data_df, updated_seq_df)
        updated_seq_df = calculate_3m_reaction_time(ball_data_df, updated_seq_df)

        new_file_name = os.path.splitext(sequence_file)[0] + '_new.csv'
        updated_seq_df.to_csv(new_file_name, index=False, sep=';')
        print(f"Processed file: {new_file_name}, Filtered DataFrame size: {updated_seq_df.shape}")

        # Generate and save the plots
        plot_acceleration_and_angle_change(ball_data_df, updated_seq_df, sequence_file)

# Execute the main function
if __name__ == "__main__":
    main(folder_path, ball_files_directory)