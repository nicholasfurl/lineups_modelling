# %%
##############
def get_matching_trials(face_descriptions,gender):

    #I will always get 40 perps
    num_perps = 40  #First face in pair, so this is also number of trials
    
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

    
    
    # #organise males and females into one big list, with matches (first half) half male and the mismatches (second half) half male
    # identity_nums_perps = np.concatenate(
    #     (
    #     identity_nums_perps_male[int(identity_nums_perps_male.shape[0]/2)], 
    #     identity_nums_perps_female[int(identity_nums_perps_female.shape[0]/2)],
    #     identity_nums_perps_male[int(identity_nums_perps_male.shape[0]/2):2*int(identity_nums_perps_male.shape[0]/2)], 
    #     identity_nums_perps_female[int(identity_nums_perps_female.shape[0]/2):2*int(identity_nums_perps_female.shape[0]/2)]
    #     )
    #     )
    
    # identity_nums_all = face_descriptions['Identity'].unique() # Select all unique identities (array)
    # # identity_nums_for_this_run = unique_identities[np.random.choice(identity_nums_all.shape[0], size=num_perps+num_fillers, replace=False)]
    
    # #identity_nums are the identity labels in the identity column of face_descriptions. They are not the indices into the rows of face_descriptions
    # identity_nums_perps = identity_nums_all[np.random.choice(identity_nums_all.shape[0], size=num_perps, replace=False)]
    
    # #Get indices into face descriptions of the perp and filler stimuli with correct car levels and expressions 
    paired = [] #information about the face image paired with every perp in each trial
    
    car_conds = {
        1: "Veridical",
        2: "Caricature",
        3: "Anticaricature"
        }
    
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
            (face_descriptions["Caricature level"] == "Veridical")]
            
        #assign that row to perps
        perps.append( fd_row.iloc[[0]].copy() )
        
        #Set that index as used in original face_descriptions dataframe so it won't get selected as a filler later
        #face_descriptions.loc[fd_row.index,"Used"] = 1
        #face_descriptions.loc[face_descriptions["Identity"] == this_identity,"Used" ]=1 #set all rows with this identity to 1 (and nothing else!)
        this_identity = perps[count].iloc[0]["Identity"]   #which identity was just selected Ppython subsetting is embarrassing inelegant!!!!)?
        face_descriptions.loc[face_descriptions["Identity"] == this_identity,"Used" ]=1 #set all rows with this identity to 1 (and nothing else!)
        
        
    perps = pd.concat(perps)    #Collapse the array of dataframes down into one big dataframe
        
    
    #Now do a second loop where you build up a dataframe of faces paired to the perps,
    #excluding the perps (Used == 1) and excluding paired's as we go ...
    paired = []
    for count, identity in enumerate(identity_nums_perps):  # Iterate through identity nums of randomly-sampled perps
    
        ###Get second image in pair for this trial's perp .....
        
        #which expression should this paired be?
        paired_expression = "Neutral"
        if (perps.iloc[count]["Expression"] == "Neutral"): paired_expression = "Smiling" #If perp on this trial is neutral, switch to smiling 
        
        #Which car level should paired be?
        this_car_cond = car_conds[((count+1) - 1) % 3 + 1]  #Rotate through the car level condition of the paired face every three trials
        
        if count+1 <= int(identity_nums_perps.shape[0]/2):    #The first half of perps are matched identity trials, so paired is same identity to perp
        
            #find (should be unique) row in face_descriptions with same identity as perp, opposite expression and appropriate car level
            fd_row = []
            fd_row = face_descriptions[
                (face_descriptions["Identity"] == identity) & 
                (face_descriptions["Expression"] == paired_expression) & 
                (face_descriptions["Caricature level"] == this_car_cond)
                ]    
    
        else:   #The second half of perps are non-matched identity (filler) trials
        
            this_race = perps.iloc[count]["Race"]
            this_gender = perps.iloc[count]["Gender"]
        
            #This should get all candidate faces that have the same race, gender, opposite expression as perp, the appropriate car level and has not already been used
            fd_row = []
            fd_row = face_descriptions[
                (face_descriptions["Used"] != 1) &
                (face_descriptions["Gender"] == this_gender) &
                (face_descriptions["Race"] == this_race) & 
                (face_descriptions["Expression"] == paired_expression) & 
                (face_descriptions["Caricature level"] == this_car_cond)
                ]
            
        #Now you have fd_row, a dataframe where the rows are candidiates for a paired face, 
        #which are identities that match demographically but have unused identities.#
        #Shuffle candidte list then take the first one, to ensure no systematic / predictable pairs
        fd_row = fd_row.sample(frac=1)
        #.reset_index(drop=True)
            
        #assign the first row to paired
        try:
            paired.append( fd_row.iloc[[0]].copy() )
        
        except Exception as e:
            print('')
            
        
        
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
    sns.scatterplot(x=flat_space[:, 0], y=flat_space[:, 1], hue=face_descriptions['Caricature level'])
    
    #plot expressions
    plt.figure()
    sns.scatterplot(x=flat_space[:, 0], y=flat_space[:, 1], hue=face_descriptions['Expression'])
    
    #plot identities
    plt.figure()
    sns.scatterplot(x=flat_space[:, 0], y=flat_space[:, 1], hue=face_descriptions['Identity'])
  ##############################










# %%
##########################
#Analyse matching performance on the trials based on the distances
def get_trial_distances(face_descriptions, dat, gender):
    
    #Initialise dataframe to hold distances results
    perps_distances = {
        'Simulated participant': [],
        'Distance': [],
        'Match': [],
        'Caricature level': [],
    }
    perps_distances = pd.DataFrame(perps_distances)
    
    print("Creating and processing simulated participants")
    
    num_fake_subs = 50
    for fake_sub in range(num_fake_subs):
    
        #What I need to do matching are the indices into the rows of Ematrix for these perps and fillers
        perps, paired = get_matching_trials(face_descriptions,gender)
        
        #Get Euclidean distance matrix
        from scipy.spatial.distance import pdist, squareform
        
        #Note, rows and cols of this matrix should lineup exactly with rows of face_descriptions and dat and the index numbers in perps and paired
        Ematrix = squareform(pdist(np.asarray(dat),'euclidean'))
        # Ematrix = squareform(pdist(np.asarray(dat),'cityblock'))

        
        # #visualise the matrix
        # import seaborn as sns
        # p1 = sns.heatmap(Ematrix)
    
        
        #Get distance for each pair and label if objectively a match trial or not
        perps = perps.reset_index() #Move indices inherited from face_description into own column where it can be more easily accessed and make new index
        paired = paired.reset_index() #Move indices inherited from face_description into own column where it can be more easily accessed and make new index
        
            #summarise genders of output(for debugging)
        man_count = (paired['Gender'] == 'Man').sum()
        woman_count = (paired['Gender'] == 'Woman').sum()
        print(f"fake_subject: {fake_sub}, {man_count} 'Man' values, {woman_count} 'Woman' values")

        for perp_i in range(len(perps)):
            
            distance = Ematrix[
            perps.iloc[perp_i]["index"],
            paired.iloc[perp_i]["index"]
            ]
            
            match = "Mismatch"
            if perps.iloc[perp_i]["Identity"] == paired.iloc[perp_i]["Identity"]:
                match = "Match"
            
            new_row_data = {
                'Simulated participant': fake_sub,
                'Distance': distance,
                'Match': match,
                'Caricature level': paired.iloc[perp_i]["Caricature level"],
            }
        
            #The complexity of this command just to add a row in a loop to a dataframe is insane, Python (not even counting the discontinuation of append for concat)!
            # if (gender == "All") | (paired.iloc[perp_i]["Gender"] == gender):
            perps_distances = pd.concat([perps_distances,pd.DataFrame([new_row_data])], ignore_index=True)
            
            
    ####THE DISTANCES PLOT!!!!!######
    fontsize = 22
    fig, axs = plt.subplots(1, 1, figsize=(6, 7))
    
    colors = ['blue','red','green']
    customPalette = sns.set_palette(sns.color_palette(colors))
    
    hue_order = ["Anticaricature", "Veridical", "Caricature"]
    order = ['Match', 'Mismatch']
    
    # # Underlay the bar plot
    # sns.barplot(ax=axs, data=perps_distances, x='Match', hue='Caricature level', y='Distance', ci=None, order = order, hue_order = hue_order, alpha=0.3)
    
    # # Create box plots with dodging
    sns.boxplot(data=perps_distances, x='Match', y='Distance', hue='Caricature level', order = order, hue_order = hue_order, dodge=True, boxprops=dict(alpha=.2),showfliers = False)
    
    sns.stripplot(ax=axs, x='Match', y='Distance', hue='Caricature level', data=perps_distances, order = order, hue_order = hue_order, dodge=True, jitter=True )

    axs.tick_params(labelsize=fontsize)  # Increase tick label font size
    axs.set_xlabel('')  # Suppress x-axis label
    axs.legend(title='', fontsize=fontsize)  # Increase font size of legend title
    axs.set_xlabel('', fontsize=fontsize)
    axs.set_ylabel('Distance to paired face', fontsize=fontsize)
    plt.tight_layout()
    
    # Make the legend box background more transparent
    handles, labels = axs.get_legend_handles_labels()
    legend = axs.legend(handles[0:3], labels[0:3], title='', fontsize=fontsize-2)
    # legend = axs.legend(handles[0:3], labels[0:3], title='', fontsize=fontsize-2, loc='upper left', bbox_to_anchor=(0, 1), framealpha=0.3)

    # legend.get_frame().set_alpha(0.3)
    
    axs.legend_.remove()
    
    plt.show()
                

    
    # plt.figure(figsize=(4, 8))  # Adjust width and height as needed

    # # Plot the point spread plot of distances with each participant as a colored point
    # sns.stripplot(data=perps_distances, x='Match', y='Distance', hue='Caricature level', dodge=True, jitter=True)
    
    # # Create box plots with dodging
    # sns.boxplot(data=perps_distances, x='Match', y='Distance', hue='Caricature level', dodge=True, boxprops=dict(alpha=.3))
    
    # plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))  
    
    # plt.show()
    
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
        "Caricature level": car_level, 
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
    
    # auc_diag = np.trapz([0, 1],[0, 1])
    participants = ROC_data_ss['Simulated participant'].unique()
    car_levels = ["Anticaricature", "Veridical", "Caricature"]

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
                (ROC_data_ss['Match'] == "Match") &
                (ROC_data_ss['Caricature level'] == car_level)
                ]["Match responses"]
            
            this_fas_data = ROC_data_ss[
                (ROC_data_ss['Simulated participant'] == participant) & 
                (ROC_data_ss['Match'] == "Mismatch") &
                (ROC_data_ss['Caricature level'] == car_level)
                ]["Match responses"]
            
            new_row_data = {
                'Simulated participant': participant,
                'Test caricature': car_level,
                'Area under the curve (AUC)': np.trapz(this_hits_data,this_fas_data)   #careful, the argument for y comes first here!
            }
            
            AUC = pd.concat([AUC,pd.DataFrame([new_row_data])], ignore_index=True)
            
            
            
    ####THE AUC PLOT!!!!!######
    fontsize = 22
    fig, axs = plt.subplots(1, 1, figsize=(8, 6))
            
    colors = ['blue','red','green']
    customPalette = sns.set_palette(sns.color_palette(colors))
            
    # Underlay the bar plot
    sns.barplot(ax=axs, data=AUC, x='Test caricature', y='Area under the curve (AUC)', ci=None, palette=customPalette, alpha=0.3, hue_order = car_levels, order = car_levels)
 
    # Plot the point spread plot of distances with each participant as a colored point
    sns.stripplot(ax=axs,data=AUC, x='Test caricature', y='Area under the curve (AUC)', hue = "Test caricature", jitter=True, hue_order = car_levels, order = car_levels)
    
    axs.tick_params(labelsize=fontsize)  # Increase tick label font size
    axs.set_ylabel('Discriminability (AUC)', fontsize=fontsize)
    axs.set_xlabel('Test caricature', fontsize=fontsize)
    axs.set_ylim(0.49, 1.01)  # Set y-axis limits
    axs.set_yticks([0.5,.6,.7,.8,.9,1])
    axs.legend_.remove()

    plt.tight_layout()
    plt.show()  
    
    return AUC
        




# %%
##########################
def get_ROC_data(perps_distances):
    
    #Loop through criterion distances and count accuracy, hits and false alarms 
    num_criteria = 10
    criteria = np.linspace(
        np.min(perps_distances['Distance']),
        np.max(perps_distances['Distance']), 
               num_criteria)    
    
    # Initialize an empty lists to store data extracted for each criterion
    ROC_data_list = []
    ROC_data_ss_list = []
    
    for i in range(num_criteria):
        
        # Create a temporary DataFrame perps_distances_temp for each iteration
        perps_distances_temp = perps_distances.copy()
        
        # Add a new column with 1 where Distance is less than the current criteria value, and 0 otherwise
        perps_distances_temp["Match responses"] = (perps_distances_temp['Distance'] < criteria[i]).astype(int)
    
        # Compute mean for every 'Simulated_Participant', 'Caricature level', and 'Match' group
        mean_distances = perps_distances_temp.groupby(['Simulated participant', 'Match', 'Caricature level']).mean()
        
        # Compute mean and confidence interval collapsing over 'Simulated_Participant'
        mean_ci = mean_distances.groupby(['Match', 'Caricature level']).agg({'Match responses': ['mean', 'sem']})
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
def plot_ROC(ROC_data):
    
    fig, axs = plt.subplots(1, 1, figsize=(7, 7))

    colors = ['blue', 'red', 'green']
    customPalette = sns.set_palette(sns.color_palette(colors))
    
    ordered_labels = ["Anticaricature", "Veridical", "Caricature"]
    
    lines = []
    # for level in caricature_levels:
    for level, color in zip(ordered_labels,colors):

        # Filter data for the current caricature level
        data_level = ROC_data[ROC_data['Caricature level'] == level]
        
        # Separate data for 'Match' and 'Mismatch'
        match_data = data_level[data_level['Match'] == 'Match']
        mismatch_data = data_level[data_level['Match'] == 'Mismatch']
        
        # Plot ROC curve
        lines.append(axs.plot(mismatch_data['mean'], match_data['mean'], label=level, color = color, marker='o', markerfacecolor = color, markersize = 10)[0])
        
    # Draw a diagonal line where x=y
    axs.plot([0, 1], [0, 1], color='black', linewidth=2, label='Chance', linestyle='--')
        
    fontsize = 22
    # axs[0].set_title('Target present', fontsize=fontsize)
    axs.tick_params(labelsize=fontsize)  # Increase tick label font size
    axs.set_xlabel('Cumulative false alarm rate')  # Suppress x-axis label
    axs.set_ylabel('Cumulative hit rate')  # Suppress x-axis label
    axs.legend(title='', fontsize=fontsize)  # Increase font size of legend title
    axs.set_ylim(-0.05, 1.05)  # Set y-axis limits
    axs.set_xlim(-0.05, 1.05)  # Set x-axis limits
    
    # Set ticks every 0.2 units
    axs.set_xticks(np.arange(0, 1.1, 0.2))
    axs.set_yticks(np.arange(0, 1.1, 0.2))
   
    axs.set_xlabel(axs.get_xlabel(), fontsize=fontsize)
    axs.set_ylabel(axs.get_ylabel(), fontsize=fontsize)
    
    # Force the axis to be square
    axs.set_aspect('equal', 'box')
   
    axs.grid(False)
       
    # Adjust layout
    plt.tight_layout()
    
    plt.show()    
        
 









#############################
def plot_ROCs(ROC_data):
        
    # Plot ROC curves for different caricature levels
    caricature_levels = ROC_data['Caricature level'].unique()
    caricature_levels = sorted(caricature_levels, reverse=True)  # Sort in reverse alphabetical order so the levels are plotted with same colours as distances plot
    
    
    plt.figure(figsize=(10, 6))
    
    # Modify font sizes using rcParams
    plt.rcParams.update({
        'font.size': 14,          # Base font size
        'axes.labelsize': 16,     # Size of axis labels
        'axes.titlesize': 18,     # Size of plot title
        'legend.fontsize': 14,    # Size of legend text
    })
    
    plt.figure(figsize=(8, 8))  # Adjust width and height as needed
    
    for level in caricature_levels:
        
        # Filter data for the current caricature level
        data_level = ROC_data[ROC_data['Caricature level'] == level]
        
        # Separate data for 'Match' and 'Mismatch'
        match_data = data_level[data_level['Match'] == 'Match']
        mismatch_data = data_level[data_level['Match'] == 'Mismatch']
                
        # Plot ROC curve
        plt.plot(mismatch_data['mean'], match_data['mean'], label=level)
        
        # Add shaded error area
        plt.fill_between(mismatch_data['mean'], match_data['mean'] - match_data['CI'], match_data['mean'] + match_data['CI'], alpha=0.3)
    
        
    
    # Add labels and legend
    plt.xlabel('False alarm rate')
    plt.ylabel('Hit rate')
    plt.title('Matching task')
    plt.legend(title='Caricature Level')
    # plt.grid(True)
    
    # Draw a diagonal line where x=y
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    
    plt.show()
#######################################  





# %%
#######################################
def plot_distances_by_car_level(Ematrix, face_descriptions):
    
    #plot distances between caricatures, veridical and anticaricatures
    # Get the indices corresponding to each combination of Caricature level and Expression
    indices = {
        ('Neutral', 'Caricature'): face_descriptions[(face_descriptions['Expression'] == 'Neutral') & (face_descriptions['Caricature level'] == 'Caricature')].index,
        ('Smiling', 'Caricature'): face_descriptions[(face_descriptions['Expression'] == 'Smiling') & (face_descriptions['Caricature level'] == 'Caricature')].index,
        ('Neutral', 'Veridical'): face_descriptions[(face_descriptions['Expression'] == 'Neutral') & (face_descriptions['Caricature level'] == 'Veridical')].index,
        ('Smiling', 'Veridical'): face_descriptions[(face_descriptions['Expression'] == 'Smiling') & (face_descriptions['Caricature level'] == 'Veridical')].index,
        ('Neutral', 'Anticaricature'): face_descriptions[(face_descriptions['Expression'] == 'Neutral') & (face_descriptions['Caricature level'] == 'Anticaricature')].index,
        ('Smiling', 'Anticaricature'): face_descriptions[(face_descriptions['Expression'] == 'Smiling') & (face_descriptions['Caricature level'] == 'Anticaricature')].index
    }
    
    # Mask out elements above the diagonal and assign NaN
    np.fill_diagonal(Ematrix, np.nan)  # Exclude distances of each item with itself
    below_diagonal = np.tril(Ematrix, k=-1)  # Extract values below the diagonal and assign NaN above
    
    # Calculate distances for each combination
    distances = {}
    for key, idx in indices.items():
        distances[key] = Ematrix[np.ix_(idx, idx)]
       
    # Create a new DataFrame
    data = {'Distance': [], 'Expression': [], 'Caricature level': []}
    for key, dist in distances.items():
        data['Distance'].extend(dist.flatten())
        data['Expression'].extend([key[0]] * len(dist.flatten()))
        data['Caricature level'].extend([key[1]] * len(dist.flatten()))
    
    df = pd.DataFrame(data)
        
        # Plot splitplot superimposed over boxplots
    fig, ax = plt.subplots(figsize=(10, 6))
    
    caricature_order = ['Veridical', 'Caricature', 'Anticaricature']
    
    # Create boxplots
    sns.boxplot(data=df, x='Caricature level', y='Distance', dodge=True, ax=ax, linewidth=1,order=caricature_order)
    
    # Create splitplot
    sns.stripplot(data=df, x='Caricature level', y='Distance',  dodge=True, jitter=True, ax=ax, color='black', alpha=0.1, order=caricature_order)
    
    # Customize plot
    ax.set_ylabel('Distance')
    # ax.set_title('Euclidean Distance by Caricature level and Expression')
    
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
#(NOTE: Now that I've added back in some identities with scare races, this bit can fail when it fails to find any matching pair. There's no problem with the code, just rerun it)
Ematrix, perps_distances = get_trial_distances(face_descriptions, dat, gender = "All")

#plot_distances_by_car_level(Ematrix, face_descriptions)

#Compute cumulative hits and false alarms
ROC_data_ss, ROC_data = get_ROC_data(perps_distances)

plot_ROC(ROC_data)

AUC_data = compute_and_plot_AUC(ROC_data_ss)

    

print('audi5000')
