from typing import Callable
from socceraction.spadl.utils import add_names
import numpy as np
import pandas as pd
import torch
import itertools
import timeit
from rich.progress import track
from torch.utils.data import DataLoader, Subset, random_split
from unxpass.components import pass_selection, pass_success, pass_value, pass_value_custom
from unxpass.datasets import PassesDataset
import gc
import json
#Functions to get selection and evaluation criterions
def convert_pass_coords(pass_selection_surface, x_t, y_t):
    #edited to convert selection surface coords to real x and y
    p_odds = pass_selection_surface[y_t, x_t]
    y_dim, x_dim = pass_selection_surface.shape
    y_t = y_t / y_dim * 68 + 68 / y_dim / 2
    x_t = x_t / x_dim * 105 + 105 / x_dim / 2
    
    return x_t, y_t, p_odds
def convert_x(x_t, x_dim):
    return x_t / x_dim * 105 + 105 / x_dim / 2
def convert_y(y_t, y_dim):
    return y_t / y_dim * 68 + 68 / y_dim / 2

class LocationPredictions:
    def __init__(
        self,
        pass_selection_component: pass_selection.SoccerMapComponent,
        pass_success_component: pass_success.SoccerMapComponent,
        pass_value_success_offensive_component: pass_value_custom.SoccerMapComponent,
        pass_value_success_defensive_component: pass_value_custom.SoccerMapComponent,
        pass_value_fail_offensive_component: pass_value_custom.SoccerMapComponent,
        pass_value_fail_defensive_component: pass_value_custom.SoccerMapComponent
    ):
        self.pass_value_success_offensive_component = pass_value_success_offensive_component
        self.pass_value_success_defensive_component = pass_value_success_defensive_component
        self.pass_selection_component = pass_selection_component
        self.pass_success_component = pass_success_component
        self.pass_value_fail_offensive_component = pass_value_fail_offensive_component
        self.pass_value_fail_defensive_component = pass_value_fail_defensive_component
        
    def rate_all_games(self, db, dataset, summarize = True, custom_pass = None):
        print("Generating Surfaces:")
        pass_selection_surface = self.pass_selection_component.predict_surface(dataset, db = db, model_name = "sel")
        pass_success_surface = self.pass_success_component.predict_surface(dataset,db = db,model_name = "val")
        pass_value_surface_offensive_success = self.pass_value_success_offensive_component.predict_surface(dataset, db = db,model_name = "val")
        pass_value_surface_offensive_fail = self.pass_value_fail_offensive_component.predict_surface(dataset, db = db,model_name = "val")
        pass_value_surface_defensive_success = self.pass_value_success_defensive_component.predict_surface(dataset, db = db,model_name = "val")
        pass_value_surface_defensive_fail = self.pass_value_fail_defensive_component.predict_surface(dataset, db = db,model_name = "val")
        print("Finished.")
        alldf = []
        #sels = self.pass_selection_component.predict(dataset,model_name = "sel")#ensuring that the actual pass made is in the dataset
        #value_fail = self.pass_value_fail_offensive_component.predict(dataset, model_name = "val") - self.pass_value_fail_defensive_component.predict(dataset, model_name = "val")
        #value_success = self.pass_value_success_offensive_component.predict(dataset, model_name = "val") - self.pass_value_success_defensive_component.predict(dataset, model_name = "val")
        #success = self.pass_success_component.predict(dataset, model_name = "val")
        #all_ratings = pd.concat([sels, success, value_fail, value_success], axis=1).rename(columns = {0:"selection_probability", 1: "success_probability", 2:"value_fail", 3:"value_success"})
        for game in pass_selection_surface:
            #game_preds = all_ratings.loc[game]
            ending_coords = db.actions(game_id = game)[["end_x", "end_y"]].loc[game]
            #game_ogs = pd.concat([game_preds,ending_coords], axis = 1).dropna()
            game_df_actions = db.actions(game_id = game)
            game_selection = pass_selection_surface[game][next(iter(pass_selection_surface[game]))]
            x_lim, y_lim = game_selection.shape
            coords = list(itertools.product(range(0,x_lim), range(0,y_lim)))
            coords_array = np.array(coords)
            input_df = pd.DataFrame()
            input_df["coord_x"] = coords_array[:, 0]
            input_df["coord_y"] = coords_array[:, 1]
            coord_x = input_df["coord_x"].values
            coord_y = input_df["coord_y"].values
            for action in pass_selection_surface[game]:
                if (action not in pass_success_surface[game]) or (action not in pass_value_surface_offensive_success[game]):
                    print(f"Skipping:{game},{action}")
                    continue
                if custom_pass is not None:
                    game = custom_pass["game_id"]
                    action = custom_pass["action_id"]
                print(f"Writing:{game},{action}")
                row = game_df_actions.loc[(game, action)].values
                allpasses = pd.DataFrame(np.tile(row, (len(coords), 1)), columns=game_df_actions.columns).reset_index(drop=True)
                #allpasses = pd.DataFrame([game_df_actions.loc[(game, action)]] * len(coords)).reset_index(drop = True)
                input_df_act = pd.concat([allpasses, input_df], axis = 1)
                true_ends = game_df_actions.loc[(game,action)][["end_x","end_y"]].reset_index(drop = True)#this is really slow...
                input_df_act["end_x"] = coord_x / x_lim * 105 + 105 / x_lim / 2
                input_df_act["end_y"] = coord_y / y_lim * 68 + 68 / y_lim / 2
                #metrics = self.rate_optimized(game, action, pass_selection_surface, pass_success_surface, pass_value_surface_offensive_success, pass_value_surface_defensive_success,pass_value_surface_offensive_fail, pass_value_surface_defensive_fail, db, game_df_actions)
                metrics = self.rate(input_df_act, game, action, pass_selection_surface, pass_success_surface, pass_value_surface_offensive_success, pass_value_surface_defensive_success,pass_value_surface_offensive_fail, pass_value_surface_defensive_fail, db, game_df_actions)
                #print(metrics.columns) 
                #print(true_ends)               
                metrics["Dist_From_True"] = (metrics["end_x"] - true_ends[0])**2 + (metrics["end_y"] - true_ends[1])**2
                closest_idx = np.where(metrics["Dist_From_True"] == min(metrics["Dist_From_True"]))[0]#getting closest pass to actual pass and then replacing that one with the original pass
                metrics = metrics.reset_index(drop = True)#is the slowest here?
                cols = ["end_x", "end_y", "selection_probability", "success_probability", "value_success", "value_fail"]
                metrics["True_Location"] = 0
                #for col in cols:
                #    metrics[col][closest_idx[0]] = game_ogs[col].loc[action]
                metrics["selection_probability"] = np.float64(metrics["selection_probability"])
                metrics["True_Location"][closest_idx[0]] = 1
                metrics["expected_utility"] = (metrics["success_probability"] * metrics["value_success"]) + ((1 - metrics["success_probability"]) * metrics["value_fail"])
                metrics["evaluation_criterion"] = metrics["expected_utility"] - sum(metrics["selection_probability"] * metrics["expected_utility"])
                metrics["selection_criterion"] = sum(
                metrics["selection_probability"] * (metrics["evaluation_criterion"])**2
                )
                metrics = metrics.drop(columns = ["Dist_From_True"])
                if summarize:
                    metrics = metrics[metrics["True_Location"] == 1]
                    metrics = metrics[["original_event_id", "game_id", "action_id", "start_x","start_y", "end_x", "end_y", "result_id","selection_probability","success_probability","value_success_off", "value_fail_off","value_success_def", "value_fail_def","selection_criterion", "evaluation_criterion"]]
                if custom_pass is not None:
                    return metrics
                #print(metrics.columns)
                alldf.append(metrics)
            
        combined = pd.concat(alldf)
        return combined

    def rate(self, input_df, game, action, selection, success, value_success_off, value_success_def, value_fail_off, value_fail_def, db, game_df_actions):
        start_time = timeit.default_timer()
        game_selection = selection[game][action]
        game_success = success[game][action]
        game_value_success_off = value_success_off[game][action]
        game_value_success_def = value_success_def[game][action]
        game_value_fail_off = value_fail_off[game][action]
        game_value_fail_def = value_fail_def[game][action]
        #x_lim, y_lim = game_selection.shape
        #coords = list(itertools.product(range(0,x_lim), range(0,y_lim)))
        #test_db = game_df_actions
        elapsed = timeit.default_timer() - start_time
        start_time = timeit.default_timer()
        df_override = input_df #pd.DataFrame([test_db.loc[(game, action)]] * len(coords))
        df_override["game_id"] = game
        df_override["action_id"] = action
        elapsed = timeit.default_timer() - start_time
        start_time = timeit.default_timer()
        #df_override["coord_x"] = coords_array[:, 0]
        #df_override["coord_y"] = coords_array[:, 1]
        elapsed = timeit.default_timer() - start_time
        start_time = timeit.default_timer()
        #df_override["end_x"] = convert_x(df_override["coord_x"], x_lim)
        #df_override["end_y"] = convert_y(df_override["coord_y"], y_lim)
        elapsed = timeit.default_timer() - start_time
        start_time = timeit.default_timer()
        df_override["selection_probability"] = game_selection[df_override["coord_x"], df_override["coord_y"]]
        df_override["success_probability"] = game_success[df_override["coord_x"], df_override["coord_y"]]
        df_override["value_success_off"] = game_value_success_off[df_override["coord_x"], df_override["coord_y"]]
        df_override["value_fail_off"] = game_value_fail_off[df_override["coord_x"], df_override["coord_y"]]
        df_override["value_success_def"] = game_value_success_def[df_override["coord_x"], df_override["coord_y"]]
        df_override["value_fail_def"] = game_value_fail_def[df_override["coord_x"], df_override["coord_y"]]
        df_override["value_success"] = df_override["value_success_off"] - df_override["value_success_def"]
        df_override["value_fail"] = df_override["value_fail_off"] - df_override["value_fail_def"]
        elapsed = timeit.default_timer() - start_time
        start_time = timeit.default_timer()
        return df_override
    def generate_surfaces(self, output_dir, dataset, db):
        pass_selection_surface = self.pass_selection_component.predict_surface(dataset, db = db, model_name = "sel")
        with open(f"{output_dir} / selections.json", "w") as outfile: 
            json.dump(pass_selection_surface, outfile)
        del pass_selection_surface
        gc.collect()
        pass_success_surface = self.pass_success_component.predict_surface(dataset,db = db,model_name = "val")
        with open(f"{output_dir} / success.json", "w") as outfile: 
            json.dump(pass_success_surface, outfile)
        del pass_success_surface
        gc.collect()
        pass_value_surface_offensive_success = self.pass_value_success_offensive_component.predict_surface(dataset, db = db,model_name = "val")
        with open(f"{output_dir} / value_off_success.json", "w") as outfile: 
            json.dump(pass_sn_surface, outfile)
        del pass_value_surface_offensive_success
        gc.collect()
        pass_value_surface_offensive_fail = self.pass_value_fail_offensive_component.predict_surface(dataset, db = db,model_name = "val")
        with open(f"{output_dir} / value_off_fail.json", "w") as outfile: 
            json.dump(pass_selection_surface, outfile)
        del pass_value_surface_offensive_fail
        gc.collect()
        pass_value_surface_defensive_success = self.pass_value_success_defensive_component.predict_surface(dataset, db = db,model_name = "val")
        with open(f"{output_dir} / value_def_success.json", "w") as outfile: 
            json.dump(pass_selection_surface, outfile)
        del pass_value_surface_defensive_success
        gc.collect()
        pass_value_surface_defensive_fail = self.pass_value_fail_defensive_component.predict_surface(dataset, db = db,model_name = "val")
        with open(f"{output_dir} / value_def_fail.json", "w") as outfile: 
            json.dump(pass_selection_surface, outfile)
        del pass_value_surface_defensive_fail
        gc.collect()
    def rate_one_game(self, db, dataset, game_id, summarize = True, custom_pass = None):
        print("Generating Surfaces:")
        pass_selection_surface = self.pass_selection_component.predict_surface(dataset, db = db, model_name = "sel", game_id = game_id)
        pass_success_surface = self.pass_success_component.predict_surface(dataset,db = db,model_name = "val", game_id = game_id)
        pass_value_surface_offensive_success = self.pass_value_success_offensive_component.predict_surface(dataset, db = db,model_name = "val", game_id = game_id)
        pass_value_surface_offensive_fail = self.pass_value_fail_offensive_component.predict_surface(dataset, db = db,model_name = "val", game_id = game_id)
        pass_value_surface_defensive_success = self.pass_value_success_defensive_component.predict_surface(dataset, db = db,model_name = "val", game_id = game_id)
        pass_value_surface_defensive_fail = self.pass_value_fail_defensive_component.predict_surface(dataset, db = db,model_name = "val", game_id = game_id)
        print("Finished.")
        alldf = []
        #sels = self.pass_selection_component.predict(dataset,model_name = "sel")#ensuring that the actual pass made is in the dataset
        #value_fail = self.pass_value_fail_offensive_component.predict(dataset, model_name = "val") - self.pass_value_fail_defensive_component.predict(dataset, model_name = "val")
        #value_success = self.pass_value_success_offensive_component.predict(dataset, model_name = "val") - self.pass_value_success_defensive_component.predict(dataset, model_name = "val")
        #success = self.pass_success_component.predict(dataset, model_name = "val")
        #all_ratings = pd.concat([sels, success, value_fail, value_success], axis=1).rename(columns = {0:"selection_probability", 1: "success_probability", 2:"value_fail", 3:"value_success"})
        for game in pass_selection_surface:
            #game_preds = all_ratings.loc[game]
            ending_coords = db.actions(game_id = game)[["end_x", "end_y"]].loc[game]
            #game_ogs = pd.concat([game_preds,ending_coords], axis = 1).dropna()
            game_df_actions = db.actions(game_id = game)
            game_selection = pass_selection_surface[game][next(iter(pass_selection_surface[game]))]
            x_lim, y_lim = game_selection.shape
            coords = list(itertools.product(range(0,x_lim), range(0,y_lim)))
            coords_array = np.array(coords)
            input_df = pd.DataFrame()
            input_df["coord_x"] = coords_array[:, 0]
            input_df["coord_y"] = coords_array[:, 1]
            coord_x = input_df["coord_x"].values
            coord_y = input_df["coord_y"].values
            for action in pass_selection_surface[game]:
                if (action not in pass_success_surface[game]) or (action not in pass_value_surface_offensive_success[game]):
                    print(f"Skipping:{game},{action}")
                    continue
                if custom_pass is not None:
                    game = custom_pass["game_id"]
                    action = custom_pass["action_id"]
                print(f"Writing:{game},{action}")
                row = game_df_actions.loc[(game, action)].values
                allpasses = pd.DataFrame(np.tile(row, (len(coords), 1)), columns=game_df_actions.columns).reset_index(drop=True)
                #allpasses = pd.DataFrame([game_df_actions.loc[(game, action)]] * len(coords)).reset_index(drop = True)
                input_df_act = pd.concat([allpasses, input_df], axis = 1)
                true_ends = game_df_actions.loc[(game,action)][["end_x","end_y"]].reset_index(drop = True)#this is really slow...
                input_df_act["end_x"] = coord_x / x_lim * 105 + 105 / x_lim / 2
                input_df_act["end_y"] = coord_y / y_lim * 68 + 68 / y_lim / 2
                #metrics = self.rate_optimized(game, action, pass_selection_surface, pass_success_surface, pass_value_surface_offensive_success, pass_value_surface_defensive_success,pass_value_surface_offensive_fail, pass_value_surface_defensive_fail, db, game_df_actions)
                metrics = self.rate(input_df_act, game, action, pass_selection_surface, pass_success_surface, pass_value_surface_offensive_success, pass_value_surface_defensive_success,pass_value_surface_offensive_fail, pass_value_surface_defensive_fail, db, game_df_actions)
                #print(metrics.columns) 
                #print(true_ends)               
                metrics["Dist_From_True"] = (metrics["end_x"] - true_ends[0])**2 + (metrics["end_y"] - true_ends[1])**2
                closest_idx = np.where(metrics["Dist_From_True"] == min(metrics["Dist_From_True"]))[0]#getting closest pass to actual pass and then replacing that one with the original pass
                metrics = metrics.reset_index(drop = True)#is the slowest here?
                cols = ["end_x", "end_y", "selection_probability", "success_probability", "value_success", "value_fail"]
                metrics["True_Location"] = 0
                #for col in cols:
                #    metrics[col][closest_idx[0]] = game_ogs[col].loc[action]
                metrics["selection_probability"] = np.float64(metrics["selection_probability"])
                metrics["True_Location"][closest_idx[0]] = 1
                metrics["expected_utility"] = (metrics["success_probability"] * metrics["value_success"]) + ((1 - metrics["success_probability"]) * metrics["value_fail"])
                metrics["evaluation_criterion"] = metrics["expected_utility"] - sum(metrics["selection_probability"] * metrics["expected_utility"])
                metrics["selection_criterion"] = sum(
                metrics["selection_probability"] * (metrics["evaluation_criterion"])**2
                )
                metrics = metrics.drop(columns = ["Dist_From_True"])
                if summarize:
                    metrics = metrics[metrics["True_Location"] == 1]
                    metrics = metrics[["original_event_id", "game_id", "action_id", "start_x","start_y", "end_x", "end_y", "result_id","selection_probability","success_probability","value_success_off", "value_fail_off","value_success_def", "value_fail_def","selection_criterion", "evaluation_criterion"]]
                if custom_pass is not None:
                    return metrics
                #print(metrics.columns)
                alldf.append(metrics)
            
        combined = pd.concat(alldf)
        return combined

