import os
import yaml
import pandas as pd

# Paths
input_dir = 'data/ipl_yaml/'
output_csv = 'outputs/ipl_matches.csv'

records = []

for file in os.listdir(input_dir):
    if file.endswith(".yaml"):
        with open(os.path.join(input_dir, file), 'r') as f:
            try:
                match_data = yaml.safe_load(f)
                info = match_data.get('info', {})
                innings = match_data.get('innings', [])

                for inning in innings:
                    for team, deliveries in inning.items():
                        for delivery in deliveries['deliveries']:
                            for ball_num, ball_info in delivery.items():

                                # Safe over/ball extraction
                                if isinstance(ball_num, float):
                                    over = int(ball_num)
                                    ball = int(round((ball_num - over) * 10))
                                else:
                                    over = ball_num[0]
                                    ball = ball_num[1]

                                runs_info = ball_info.get('runs', {})

                                record = {
                                    'match_id': file.replace('.yaml', ''),
                                    'batting_team': team,
                                    'over': over,
                                    'ball': ball,
                                    'batter': ball_info.get('batter') or ball_info.get('batsman'),
                                    'bowler': ball_info.get('bowler'),
                                    'non_striker': ball_info.get('non_striker') or ball_info.get('non-striker'),
                                    'runs_batter': runs_info.get('batter') or runs_info.get('batsman'),
                                    'runs_total': runs_info.get('total'),
                                    'runs_extras': runs_info.get('extras'),
                                    'wickets': len(ball_info.get('wickets', [])),
                                    'venue': info.get('venue'),
                                    'team1': info.get('teams', [None, None])[0],
                                    'team2': info.get('teams', [None, None])[1],
                                    'date': info.get('dates', [None])[0]
                                }
                                records.append(record)
            except Exception as e:
                print(f"⚠️ Error processing file {file}: {e}")

# Save to CSV
df = pd.DataFrame(records)
df.to_csv(output_csv, index=False)
print(f"✅ Saved {len(df)} ball-by-ball records to {output_csv}")
