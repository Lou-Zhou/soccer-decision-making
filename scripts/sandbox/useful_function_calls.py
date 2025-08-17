
import sdm.visualization


sdm.visualization.plot_model_outputs(
    component = "selection",
    run_id = "cb051b26ef7640f9834e360fb3ca0c1b",
    path_feature = "Hawkeye/Hawkeye_Features/sequences",
    game_id = 3835320,
    show_pass = False
)   # takes about a minute

sdm.visualization.generate_pass_surface_gifs(
    component = "selection",
    run_id = "cb051b26ef7640f9834e360fb3ca0c1b",
    path_feature = "Hawkeye/Hawkeye_Features/sequences_tenSecPrior",
    path_play = "steffen/sequence_filtered.csv",
    game_id = 3835320,
)   # takes a few hours
