#####
#This one does old / new task when caricature is manipulated at test ONLY 
#(i.e., as it actually was in RQ1). 

#I'll do manipulation of caricature at both study and test in a separate program



# %%
##############
def get_oldnew_trials(face_descriptions,gender):
    
    #perps means old facial identities
    num_perps = 20  #First face in pair, so this is also number of trials
    

    if (gender == "All"):
        num_male_perps = int(num_perps/2)
        num_female_perps = int(num_perps/2)
    elif (gender == "Man"):
        num_male_perps = int(num_perps)
        num_female_perps = 0
    elif (gender == "Woman"):
        num_male_perps = 0
        num_female_perps = int(num_perps)

    # Randomly sample male identities for half of the perps
    identity_nums_male = face_descriptions[face_descriptions['Gender'] == 'Man']['Identity'].unique()
    identity_nums_perps_male = identity_nums_male[np.random.choice(identity_nums_male.shape[0], size=num_male_perps, replace=False)]
    
    # Randomly sample female identities for other half of the perps
    identity_nums_female = face_descriptions[face_descriptions['Gender'] == 'Woman']['Identity'].unique()
    identity_nums_perps_female = identity_nums_female[np.random.choice(identity_nums_female.shape[0], size=num_female_perps, replace=False)]
    
    #In case males concatenated with females, redistribute so equal genders for matches and mismatches
    identity_nums_perps = np.concatenate((identity_nums_perps_male, identity_nums_perps_female))
        
        # Split the array into quarters
    quarter_size = len(identity_nums_perps) // 4
    first_quarter = identity_nums_perps[:quarter_size]
    second_quarter = identity_nums_perps[quarter_size:2*quarter_size]
    third_quarter = identity_nums_perps[2*quarter_size:3*quarter_size]
    fourth_quarter = identity_nums_perps[3*quarter_size:]
    
    # Reorganize the second and third quarters
    identity_nums_perps = np.concatenate((first_quarter, third_quarter, second_quarter, fourth_quarter))
        
   

    #It's inelegent, but I'm going to do one loop for the perps first, so that I can
    #mark them all as used before I then do the paired faces in a second loop
    perps = []  #information about every perp
    face_descriptions = face_descriptions.assign(Used=0)
    for count, identity in enumerate(identity_nums_perps):  # Iterate through identity nums of randomly-sampled perps
        
        ###get version for this trial of perp identity .....
        
        #which expression should perp be?
        this_expression = "Neutral"
        if (count % 2 == 0): this_expression = "Smiling"
        
        #find (should be unique) row in face_descriptions with this identity and expression and Veridical car level
        fd_row = []
        fd_row = face_descriptions[
            (face_descriptions["Identity"] == identity) & 
            (face_descriptions["Expression"] == this_expression) & 
            (face_descriptions["Test caricature"] == "Veridical")]
            
        #assign that row to perps
        perps.append( fd_row.iloc[[0]].copy() )
        
        #Set that index as used in original face_descriptions dataframe so it won't get selected as a filler later
        #face_descriptions.loc[fd_row.index,"Used"] = 1
        #face_descriptions.loc[face_descriptions["Identity"] == this_identity,"Used" ]=1 #set all rows with this identity to 1 (and nothing else!)
        this_identity = perps[count].iloc[0]["Identity"]   #which identity was just selected Ppython subsetting is embarrassing inelegant!!!!)?
        face_descriptions.loc[face_descriptions["Identity"] == this_identity,"Used" ]=1 #set all rows with this identity to 1 (and nothing else!)
        
        
    perps = pd.concat(perps)    #Collapse the array of dataframes down into one big dataframe
       
    
    #prepare to define the test faces (paired dataframe)
    paired = []
    car_conds = {
        1: "Veridical",
        2: "Caricature",
        3: "Anticaricature"
        }
    
    #Now using second loop get test versions of the study faces (old faces) and slot them into the first half of the rows in paired 
    for count, identity in enumerate(identity_nums_perps):  # Iterate through identity nums of randomly-sampled perps
        
        #which expression should this paired be?
        paired_expression = "Neutral"
        if (perps.iloc[count]["Expression"] == "Neutral"): paired_expression = "Smiling" #If perp on this trial is neutral, switch to smiling 
        
        #Which car level should paired be?
        this_car_cond = car_conds[((count+1) - 1) % 3 + 1]  #Rotate through the car level condition of the paired face every three trials
        
        #find (should be unique) row in face_descriptions with same identity as perp, opposite expression and appropriate car level
        fd_row = []
        fd_row = face_descriptions[
            (face_descriptions["Identity"] == identity) & 
            (face_descriptions["Expression"] == paired_expression) & 
            (face_descriptions["Test caricature"] == this_car_cond)
            ] 
        
        #assign a copy of first row to paired
        fd_row_copy = fd_row.iloc[0:1].copy()
        fd_row_copy['Type'] = 'Old'
        paired.append( fd_row_copy )
        
        
            
#Now using third loop get test faces matched to perp list characteristics (new faces) and slot them into the second half of the rows in paired 
    for count, identity in enumerate(identity_nums_perps):  # Iterate through identity nums of randomly-sampled perps
        
            #Match the race and gender of this new face to one of the perps
            this_race = perps.iloc[count]["Race"]
            this_gender = perps.iloc[count]["Gender"]
            
            #which expression should this paired be?
            paired_expression = "Neutral"
            if (perps.iloc[count]["Expression"] == "Neutral"): paired_expression = "Smiling" #If perp on this trial is neutral, switch to smiling 
            
            #Which car level should paired be?
            this_car_cond = car_conds[((count+1) - 1) % 3 + 1]  #Rotate through the car level condition of the paired face every three trials
        
            #This should get all candidate faces that have the same race, gender, opposite expression as perp, the appropriate car level and has not already been used
            fd_row = []
            fd_row = face_descriptions[
                (face_descriptions["Used"] != 1) &
                (face_descriptions["Gender"] == this_gender) &
                (face_descriptions["Race"] == this_race) & 
                (face_descriptions["Expression"] == paired_expression) & 
                (face_descriptions["Test caricature"] == this_car_cond)
                ]
            
        #Now you have fd_row, a dataframe where the rows are candidiates for a paired face, 
        #which are identities that match demographically but have unused identities.#
        #Shuffle candidate list then take the first one, to ensure no systematic / predictable pairs
            fd_row = fd_row.sample(frac=1)
            #.reset_index(drop=True)
            
            #assign a copy of first row to paired
            fd_row_copy = fd_row.iloc[0:1].copy()
            fd_row_copy['Type'] = 'New'
            paired.append( fd_row_copy )
            
            #Set that index as used in original face_descriptions dataframe so it won't get selected as a filler later
            #face_descriptions.loc[fd_row.index,"Used"] = 1 
            this_identity = paired[count].iloc[0]["Identity"]   #which identity was just selected?
            face_descriptions.loc[face_descriptions["Identity"] == this_identity,"Used" ]=1 #set all rows with this identity to 1 (and nothing else!)
        
    paired = pd.concat(paired)
    
    return perps, paired
#####################




# %%
###################
# Function to extract filename and split by hyphen
def extract_and_split(path):
    filename = os.path.basename(path)  # Extract filename from path
    filename_without_extension = os.path.splitext(filename)[0]  # Remove extension if needed
    parts = filename_without_extension.split('-')  # Split filename by hyphen
    return parts
########################





# %%
#######################
# Visualise category exemplars in flattened 2D (t-SNE and PCA) spaces 
def tsne_and_pca(dat,face_descriptions):
   
    #t-SNE
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42,perplexity=len(dat)-1)  #Perplexity must be less than number of samples
    flat_space_tsne = tsne.fit_transform(np.asarray(dat))    #samples (i.e. pictures) by dimensions (2)
    visualise_flat_space(flat_space_tsne, face_descriptions)

    #PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    flat_space_pca = pca.fit_transform(np.asarray(dat))
    visualise_flat_space(flat_space_pca, face_descriptions)
###############################







# %%
##########################
def visualise_flat_space(flat_space,face_descriptions):
         
    #plot races
    plt.figure()
    sns.scatterplot(x=flat_space[:, 0], y=flat_space[:, 1], hue=face_descriptions['Race'])
    
    #plot genders
    plt.figure()
    sns.scatterplot(x=flat_space[:, 0], y=flat_space[:, 1], hue=face_descriptions['Gender'])
    
    #plot caricature levels
    plt.figure()
    sns.scatterplot(x=flat_space[:, 0], y=flat_space[:, 1], hue=face_descriptions['Test caricature'])
    
    #plot expressions
    plt.figure()
    sns.scatterplot(x=flat_space[:, 0], y=flat_space[:, 1], hue=face_descriptions['Expression'])
    
    #plot identities
    plt.figure()
    sns.scatterplot(x=flat_space[:, 0], y=flat_space[:, 1], hue=face_descriptions['Identity'])
  ##############################










# %%
##########################
#Analyse oldnew performance on the trials based on the distances
def get_trial_distances(face_descriptions, dat, gender):
      
    print("Creating and processing simulated participants")

    #Get Euclidean distance matrix
    from scipy.spatial.distance import pdist, squareform
    
    #Note, rows and cols of this matrix should lineup exactly with rows of face_descriptions and dat and the index numbers in perps and paired
    # Ematrix = squareform(pdist(np.asarray(dat),'euclidean'))
    Ematrix = squareform(pdist(np.asarray(dat),'cityblock'))

    # #visualise the matrix
    # import seaborn as sns
    # p1 = sns.heatmap(Ematrix)
    
    #Initialise dataframe to hold distances results
    perps_distances = {
        'Simulated participant': [],
        'Nearest distance': [],
        'Type': [],
        'Test caricature': [],
    }
    perps_distances = pd.DataFrame(perps_distances)

    num_fake_subs = 50
    for fake_sub in range(num_fake_subs):
    
        #What I need to do oldnew are the indices into the rows of Ematrix for these perps and fillers
        perps, paired = get_oldnew_trials(face_descriptions,gender)
        
        # #Get distance for each pair and label if objectively a match trial or not
        # perps = perps.reset_index() #Move indices inherited from face_description into own column where it can be more easily accessed and make new index
        # paired = paired.reset_index() #Move indices inherited from face_description into own column where it can be more easily accessed and make new index
        
        #summarise genders of output(for debugging)
        man_count = (paired['Gender'] == 'Man').sum()
        woman_count = (paired['Gender'] == 'Woman').sum()
        print(f"fake_subject: {fake_sub}, {man_count} 'Man' test faces, {woman_count} 'Woman' test faces")

        # for perp_i in range(len(perps)):
            
            # distance = Ematrix[
            # perps.iloc[perp_i]["index"],
            # paired.iloc[perp_i]["index"]
            # ]
            
        # Iterate over each row in 'paired' to find the nearest distance
        nearest_distances = {}
        it = 0
        for idx, row in paired.iterrows():
                
            # Get the distances to all items in 'perps' using the row/column index
            distances = Ematrix[idx, perps.index]
            min_distance = np.min(distances)
                        
            # match = "Mismatch"
            # if perps.iloc[it]["Identity"] == paired.iloc[it]["Identity"]:
            #     match = "Match"
            
            new_row_data = {
                'Simulated participant': fake_sub,
                'Nearest distance': min_distance,
                'Type': row.Type,
                'Test caricature': paired.iloc[it]["Test caricature"],
            }
        
            #The complexity of this command just to add a row in a loop to a dataframe is insane, Python (not even counting the discontinuation of append for concat)!
            # if (gender == "All") | (paired.iloc[perp_i]["Gender"] == gender):
            perps_distances = pd.concat([perps_distances,pd.DataFrame([new_row_data])], ignore_index=True)
            
            it = it+1
        
    # def find_nearest_distance(item_idx):
    #    # Get the distances to all items in 'perps' using the row/column index
    #    distances = Ematrix[item_idx, perps.index]
    #    # Find the minimum distance
    #    min_distance = np.min(distances)
    #    return min_distance

    # # Iterate over each row in 'paired' to find the nearest distance
    # nearest_distances = {}
    # for idx, row in paired.iterrows():
    #     nearest_distances[idx] = find_nearest_distance(idx)

    # # Convert dictionary to DataFrame
    # nearest_distances_df = pd.DataFrame.from_dict(nearest_distances, orient='index', columns=['Nearest Distance'])
    
    distances_plot_data = perps_distances.groupby(['Test caricature','Type', 'Simulated participant'])['Nearest distance'].mean().reset_index()

    plt.figure(figsize=(4, 8))  # Adjust width and height as needed
    
    # Modify font sizes using rcParams
    plt.rcParams.update({
        'font.size': 18,          # Base font size
        'axes.labelsize': 18,     # Size of axis labels
        'axes.titlesize': 18,     # Size of plot title
        'legend.fontsize': 16,    # Size of legend text
    })
    
    colors = ["#4878CF", "#FFA500", "#6ACC65"]
    customPalette = sns.set_palette(sns.color_palette(colors))
    
    x_order = ['Old','New']
    hue_order = ["Anticaricature", "Veridical","Caricature"]

    # Plot the point spread plot of distances with each participant as a colored point
    sns.stripplot(data=distances_plot_data, x='Type', y='Nearest distance', hue='Test caricature', dodge=True, jitter=True,order = x_order, hue_order = hue_order)
    
    # Create box plots with dodging
    sns.boxplot(data=distances_plot_data, x='Type', y='Nearest distance', hue='Test caricature', dodge=True, boxprops=dict(alpha=.3), order = x_order, hue_order = hue_order, showfliers=False)
    
    handles, labels = plt.gca().get_legend_handles_labels()   # Get handles and labels for legend
    
    plt.legend(handles=handles[3:6], labels=labels[3:6], title = "Test caricature", loc='upper left',framealpha=0.5, bbox_to_anchor=(1.05, 1))
    
    #plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))  
    
    plt.show()
    
    return Ematrix, perps_distances
############





# %%
######################
def get_face_descriptions_from_files():

    #Easier to relabel identities later if identities number 1 to last for neutral and then start over 1 to last for smiling
    filenames_nm = sorted(glob.glob(r'C:\matlab_files\lineups_modelling\RQ1 stimuli - Copy\Male lineups\Neutral\*.jpg'))   
    filenames_nf = sorted(glob.glob(r'C:\matlab_files\lineups_modelling\RQ1 stimuli - Copy\Female lineups\Neutral\*.jpg')) 
    filenames_sm = sorted(glob.glob(r'C:\matlab_files\lineups_modelling\RQ1 stimuli - Copy\Male lineups\Smiling\*.jpg'))     
    filenames_sf = sorted(glob.glob(r'C:\matlab_files\lineups_modelling\RQ1 stimuli - Copy\Female lineups\Smiling\*.jpg'))
    
    filenames =  np.concatenate((filenames_nm, filenames_nf, filenames_sm,filenames_sf))
    
    #break down info in filenames
    # Apply the function to each path in the numpy array
    split_filenames = [extract_and_split(path) for path in filenames]
    
    #The data only specifies if faces are car or anticar. specify neutral manually for the rest
    for lst in split_filenames:
        # Check if the last string is neither "car" nor "anticar"
        if lst[-1] not in ["car", "anticar"]:
            # Append "veridical" to the sublist
            lst.append("veridical")
    
    #Specify gender (male=1)
    gender = np.array( ["Man"]*len(filenames_nm) + ["Woman"]*len(filenames_nf) + ["Man"]*len(filenames_sm) + ["Woman"]*len(filenames_sf) )
    
    
    #Specify expression (neutral=1)
    expression =  np.array(["Neutral"] * (len(filenames_nm)+len(filenames_nf)) + ["Smiling"] * (len(filenames_sm)+len(filenames_sf)))
    
    
    #specify race as strings W (white), B (black), E (east asian), A (west asian)
    temp = np.array([lst[1][0] for lst in split_filenames]) #drop off gender and keep race part of string
    replacement_dict = {
        "W": "Caucasian",
        "B": "Black",
        "A": "West Asian",
        "E": "East Asian"
    }
    race = np.array([replacement_dict.get(item, item) for item in temp])    #clean up the labels to make nicer plots later
    
    #specify caricature level
    temp = np.array([lst[-1] for lst in split_filenames])
    replacement_dict = {
        "veridical": "Veridical",
        "car": "Caricature",
        "anticar": "Anticaricature",
    }
    car_level = np.array([replacement_dict.get(item, item) for item in temp])    #clean up the labels to make nicer plots later
    
    #assumes all identities are sorted in groups of three (one for each caricature level)
    #assumes nms, then nfs, then sms, then sfs
    num_identities = int(len(filenames_nm+filenames_nf))
    identity_list = []
    for i in range(1, int((num_identities/3)+1)):
        identity_list.extend([i]*3)
    identity = np.concatenate((identity_list,identity_list))
    
    #make dataframe
    temp_dict = {
        "Filename": [os.path.basename(name) for name in filenames], 
        "Test caricature": car_level, 
        "Gender": gender, 
        "Expression": expression, 
        "Race": race, 
        "Identity": identity
        }
    face_descriptions = pd.DataFrame(temp_dict)
    
    return filenames, face_descriptions
###########################





# %%
############################
def compute_and_plot_AUC(ROC_data_ss):
    
    auc_diag = np.trapz([0, 1],[0, 1])
    participants = ROC_data_ss['Simulated participant'].unique()
    car_levels = ROC_data_ss['Test caricature'].unique()
    car_levels = sorted(car_levels, reverse=True)  # Sort in reverse alphabetical order so the levels are plotted with same colours as distances plot

    #Initialise dataframe to hold auc results
    AUC = {
        'Simulated participant': [],
        'Test caricature': [],
        'Area under the curve (AUC)': []
    }
    AUC = pd.DataFrame(AUC)
      
    for participant in participants:
        
        for car_level in car_levels:
        
            this_hits_data = ROC_data_ss[
                (ROC_data_ss['Simulated participant'] == participant) & 
                (ROC_data_ss['Type'] == "Old") &
                (ROC_data_ss['Test caricature'] == car_level)
                ]["Old responses"]
            
            this_fas_data = ROC_data_ss[
                (ROC_data_ss['Simulated participant'] == participant) & 
                (ROC_data_ss['Type'] == "New") &
                (ROC_data_ss['Test caricature'] == car_level)
                ]["Old responses"]
            
            new_row_data = {
                'Simulated participant': participant,
                'Test caricature': car_level,
                'Area under the curve (AUC)': np.trapz(this_hits_data,this_fas_data) - auc_diag   #careful, the argument for y comes first here!
            }
            
            AUC = pd.concat([AUC,pd.DataFrame([new_row_data])], ignore_index=True)
            
    plt.figure(figsize=(4, 8))  # Adjust width and height as needed
    
    caricature_order = [ 'Anticaricature', 'Veridical', 'Caricature']
    colors = ["#4878CF", "#FFA500", "#6ACC65"]
    customPalette = sns.set_palette(sns.color_palette(colors))
         
    # Plot the point spread plot of distances with each participant as a colored point
    # sns.stripplot(data=AUC, x='Test caricature', y='Area under the curve (AUC)', hue = "Test caricature", jitter=True, dodge = .2, order = caricature_order)
    sns.stripplot(data=AUC, y = "Area under the curve (AUC)", x='Test caricature', hue='Test caricature', jitter=True, dodge=True, order = caricature_order, hue_order = caricature_order)

    # Create box plots with dodging
    # sns.boxplot(data=AUC, x='Test caricature', y='Area under the curve (AUC)', hue='Test caricature', boxprops=dict(alpha=.3), dodge = True, order = caricature_order)
    sns.boxplot(data=AUC, y = "Area under the curve (AUC)", x='Test caricature', hue='Test caricature', dodge=True, order = caricature_order, hue_order = caricature_order, showfliers=False, boxprops=dict(alpha=.3))
     
    # plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))  
    plt.ylim(0, 0.55)     # Set y-axis limits
    plt.xticks(rotation=45)  # Adjust rotation angle as needed
    
    handles, labels = plt.gca().get_legend_handles_labels()  
    
    plt.legend(handles=handles[3:6], labels=labels[3:6], title = "Test caricature", loc='upper left',framealpha=0.5, bbox_to_anchor=(1.05, 1)) 

    plt.show()   
    
    return AUC
        




# %%
##########################
def get_ROC_data(perps_distances):
    
    #Loop through criterion distances and count accuracy, hits and false alarms 
    num_criteria = 10
    criteria = np.linspace(
        np.min(perps_distances['Nearest distance']),
        np.max(perps_distances['Nearest distance']), 
               num_criteria)    
    
    # Initialize an empty lists to store data extracted for each criterion
    ROC_data_list = []
    ROC_data_ss_list = []
    
    for i in range(num_criteria):
        
        # Create a temporary DataFrame perps_distances_temp for each iteration
        perps_distances_temp = perps_distances.copy()
        
        # Add a new column with 1 where Distance is less than the current criteria value, and 0 otherwise
        perps_distances_temp["Old responses"] = (perps_distances_temp['Nearest distance'] < criteria[i]).astype(int)
    
        # Compute mean for every 'Simulated_Participant', 'Test caricature', and 'Match' group
        mean_distances = perps_distances_temp.groupby(['Simulated participant', 'Type', 'Test caricature']).mean()
        
        # Compute mean and confidence interval collapsing over 'Simulated_Participant'
        mean_ci = mean_distances.groupby(['Type', 'Test caricature']).agg({'Old responses': ['mean', 'sem']})
        mean_ci.columns = mean_ci.columns.droplevel(0)  # Remove the 'Distance' label from the column index
        
        # Calculate 95% confidence interval (using a multiplier of 1.96)
        mean_ci['CI'] = 1.96 * mean_ci['sem']
        
        # Add 'criteria' column to mean_ci DataFrame
        mean_ci['criteria'] = criteria[i]
        
        # Append mean_distances DataFrame (retaining individual fake sub means) to the list
        ROC_data_ss_list.append(mean_distances)
        
        # Append mean_ci DataFrame (collapsed over fake subs) to the list
        ROC_data_list.append(mean_ci)
        
    # Concatenate all DataFrames in the list along with the 'criteria' column (need later for AUC computation)
    ROC_data_ss = pd.concat(ROC_data_ss_list).reset_index()
        
    # Concatenate all mean_ci DataFrames in the list along with the 'criteria' column (need later for ROC plots)
    ROC_data = pd.concat(ROC_data_list).reset_index()
    
    return ROC_data_ss, ROC_data  
###############





# %%
#############################
def plot_ROCs(ROC_data):
        
    # Plot ROC curves for different caricature levels
    caricature_levels = ROC_data['Test caricature'].unique()
    caricature_levels = sorted(caricature_levels, reverse=True)  # Sort in reverse alphabetical order so the levels are plotted with same colours as distances plot
    
    
    plt.figure(figsize=(10, 6))
    
    # Modify font sizes using rcParams
    plt.rcParams.update({
        'font.size': 20,          # Base font size
        'axes.labelsize': 20,     # Size of axis labels
        'axes.titlesize': 20,     # Size of plot title
        'legend.fontsize': 20,    # Size of legend text
    })
    
    # colors = ["#4878CF", "#FFA500", "#6ACC65"]
    # customPalette = sns.set_palette(sns.color_palette(colors))
    
    # order = ["Anticaricature", "Veridical", "Caricature"]
    
    
    
    plt.figure(figsize=(8, 8))  # Adjust width and height as needed
    
    colors = ['blue', 'green', 'orange']
    ordered_labels = ["Anticaricature", "Veridical", "Caricature"]
    plt.gca().set_prop_cycle(color=colors)
    
    lines = []
    # for level in caricature_levels:
    for level, color in zip(ordered_labels,colors):

        
        # Filter data for the current caricature level
        data_level = ROC_data[ROC_data['Test caricature'] == level]
        
        # Separate data for 'Match' and 'Mismatch'
        match_data = data_level[data_level['Type'] == 'Old']
        mismatch_data = data_level[data_level['Type'] == 'New']
                
        # Plot ROC curve
        lines.append(plt.plot(mismatch_data['mean'], match_data['mean'], label=level, color = color)[0])
        
        # Add shaded error area
        plt.fill_between(mismatch_data['mean'], match_data['mean'] - match_data['CI'], match_data['mean'] + match_data['CI'], alpha=0.3)
    
        
    
    # Add labels and legend
    plt.xlabel('False alarm rate')
    plt.ylabel('Hit rate')
    # plt.title('oldnew task')
    plt.legend(title='Test caricature')
    # plt.grid(True)
    
    # ordered_labels = ["Anticaricature", "Veridical", "Caricature"]
    # ordered_lines = [lines[ordered_labels.index(label)] for label in ordered_labels]
    # plt.legend(ordered_lines, ordered_labels)
    
    
    # # Get current handles and labels
    # handles, labels = ax.get_legend_handles_labels()
    # ordered_handles = [handles[labels.index(label)] for label in ordered_labels]
    # plt.legend(ordered_handles, ordered_labels)

    
    # Draw a diagonal line where x=y
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    
    plt.show()
#######################################  





# %%
#######################################
def plot_distances_by_car_level(Ematrix, face_descriptions):
    
    #plot distances between caricatures, veridical and anticaricatures
    # Get the indices corresponding to each combination of Test caricature and Expression
    indices = {
        ('Neutral', 'Caricature'): face_descriptions[(face_descriptions['Expression'] == 'Neutral') & (face_descriptions['Test caricature'] == 'Caricature')].index,
        ('Smiling', 'Caricature'): face_descriptions[(face_descriptions['Expression'] == 'Smiling') & (face_descriptions['Test caricature'] == 'Caricature')].index,
        ('Neutral', 'Veridical'): face_descriptions[(face_descriptions['Expression'] == 'Neutral') & (face_descriptions['Test caricature'] == 'Veridical')].index,
        ('Smiling', 'Veridical'): face_descriptions[(face_descriptions['Expression'] == 'Smiling') & (face_descriptions['Test caricature'] == 'Veridical')].index,
        ('Neutral', 'Anticaricature'): face_descriptions[(face_descriptions['Expression'] == 'Neutral') & (face_descriptions['Test caricature'] == 'Anticaricature')].index,
        ('Smiling', 'Anticaricature'): face_descriptions[(face_descriptions['Expression'] == 'Smiling') & (face_descriptions['Test caricature'] == 'Anticaricature')].index
    }
    
    Ematrix_temp = Ematrix.copy()
    Ematrix_temp[np.triu_indices_from(Ematrix_temp)] = np.nan    # Set upper triangle and diagonal to NaN
    
    # Calculate distances for each combination
    distances = {}
    for key, idx in indices.items():
        distances[key] = Ematrix_temp[np.ix_(idx, idx)]
       
    # Create a new DataFrame
    data = {'Distance': [], 'Expression': [], 'Test caricature': []}
    for key, dist in distances.items():
        data['Distance'].extend(dist.flatten())
        data['Expression'].extend([key[0]] * len(dist.flatten()))
        data['Test caricature'].extend([key[1]] * len(dist.flatten()))
    
    df = pd.DataFrame(data)
        
        # Plot splitplot superimposed over boxplots
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ["#4878CF",  "#6ACC65", "#FFA500"]
    customPalette = sns.set_palette(sns.color_palette(colors))
    
    # Modify font sizes using rcParams
    plt.rcParams.update({
        'font.size': 22,          # Base font size
        'axes.labelsize': 22,     # Size of axis labels
        'axes.titlesize': 22,     # Size of plot title
        'legend.fontsize': 22,    # Size of legend text
    })
    
    caricature_order = [ 'Anticaricature', 'Veridical', 'Caricature']
    
    # Create boxplots
    sns.boxplot(data=df, x='Test caricature', y='Distance', dodge=True, ax=ax, linewidth=1,order=caricature_order, showfliers=False, palette = customPalette)
    
    # Create splitplot
    sns.stripplot(data=df, x='Test caricature', y='Distance',  dodge=True, jitter=True, ax=ax, color='black', alpha=0.1, order=caricature_order)
    
    # Customize plot
    ax.set_ylabel('Distance')
    # ax.set_title('Euclidean Distance by Test caricature and Expression')
    
    plt.tight_layout()
    plt.show()
#######################################




  

# %%
#################################################################################
###MAIN CODE BODY######################

######get RQ1 faces and construct fully described dataset

import seaborn as sns
import matplotlib.pyplot as plt
import glob 
import numpy as np
import os
import pandas as pd

pd.set_option('display.max_columns', None)

#I get a strange crash of Python (I think when I use seaborn) without this stop gap command 
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#Open files and asewmble data into primary dataframe
filenames, face_descriptions = get_face_descriptions_from_files()



#########Set up vgg16 model
# from keras.applications import vgg16
# from keras.models import Model
# model = vgg16.VGG16(weights='imagenet', include_top=True)
# model2 = Model(model.input, model.layers[-2].output)
# from keras.applications.vgg16 import preprocess_input    

#########Set up vggFACE model
from keras_vggface.vggface import VGGFace
model2 = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg')
from keras_vggface.utils import preprocess_input


#########Proprocess images and project them 
import keras.utils as image
from keras.models import Model

dat = []
imgs = []
for count, imgf in enumerate(filenames):    #CAREFUL! I used the original variable name here instead of the dataframe column because I wanted to remove the paths from the dataframe. Because they don't necessarily match without care, could cause a bug to keep using each for different things.
    
    #import image
    img = image.load_img(imgf, target_size=(224,224))
    imgs.append(img)
    img_arr = np.expand_dims(image.img_to_array(img), axis=0)

#    The images are converted from RGB to BGR, then each color channel is 
#    zero-centered with respect to the ImageNet dataset, without scaling.
#    x continues to be 1*224*224*3
    x = preprocess_input(img_arr)
    
    #preds is 1*4096
    preds = model2.predict(x) #requires "from keras.models import Model" with capital M
    
    #So we sould end up with dat, a list with one slice / item per image
    dat.append(preds[0])
    
# # Visualise category exemplars in flattened 2D (t-SNE and PCA) spaces 
# tsne_and_pca(dat,face_descriptions)

#Get distances in space for the trials and plot them by caricature
# perps_distances = get_trial_distances(face_descriptions, dat, gender = "All")
# perps_distances = get_trial_distances(face_descriptions, dat, gender = "Man")
Ematrix, perps_distances = get_trial_distances(face_descriptions, dat, gender = "All")

plot_distances_by_car_level(Ematrix, face_descriptions)

#Compute cumulative hits and false alarms
ROC_data_ss, ROC_data = get_ROC_data(perps_distances)

plot_ROCs(ROC_data)

AUC_data = compute_and_plot_AUC(ROC_data_ss)

    

print('audi5000')
