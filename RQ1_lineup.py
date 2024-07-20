#####

#I cleaned up the code, especially the figures for RQ1_oldnew and the way the old and new trials are created and saved it as RQ1_lineups.

#This one does old / new task when caricature is manipulated at test ONLY 
#(i.e., as it actually was in RQ1). 

#I'll do manipulation of caricature at both study and test in a separate program



# %%
##############
def get_oldnew_trials(face_descriptions,gender):
    
    #I started off using Gaia's two counterbalanced study and test lists. But then I changed my mind. I kepther list of perps/suspects, as these are needed to anchor a set of lineups properly matched with fillers on race, gender etc. But rather than using the same two lists for study faces (perps), I now use the set list of perps/suspect to create the lineups, then I randomly select half to be target present. And then I create a perps list from the target present lineups. So at least fake subs get a slightly different study list every time and there will be variability in the plots.
    
    #I'm going to abandon race matching the new items. I think with enough randomisations over participants this will average out, I'm not 100% sure we have enough faces for this never to produce an error when we run out of a gender / race combo and I don't know if this was done in behavioural study or not
    

    num_fillers = 4 
      
    #There are 21 "old" perps (one of these counterbalances)! So there should be 42 lineups when doubled.
    ids_fixed_cb1 = ['CFD-WM-003-006-HC.jpg','CFD-WM-033-006-HC.jpg','CFD-WM-022-003-HC.jpg','CFD-BM-002-013-N.jpg','CFD-BM-025-035-N.jpg','CFD-BM-030-003-N.jpg','CFD-WM-103_08.jpg','CFD-WM-128_08.jpg','CFD-WM-018_08.jpg','CFD-BM-114_03.jpg','CFD-E-067_03.jpg','CFD-A-037_03.jpg','CFD-WF-001-010-HC.jpg','CFD-WF-015-023-HC.jpg','CFD-WF-007-005-HC.jpg','CFD-BF-004-009-HO.jpg','CFD-BF-006-008-HC.jpg','CFD-BF-035-027-HO.jpg','CFD-BF-048-002-N.jpg','CFD-BF-031-002-N.jpg','CFD-WF-013_03.jpg']
        
    ids_fixed_cb2 = ['CFD-WM-034-035-HO.jpg','CFD-WM-026-006-HO.jpg','CFD-BM-010-004-HO.jpg','CFD-BM-039-029-N.jpg','CFD-BM-033-003-N.jpg','CFD-BM-023-029-N.jpg','CFD-WM-105_03.jpg','CFD-WM-041_03.jpg','CFD-WM-069_03.jpg','CFD-WF-009-006-HC.jpg','CFD-WF-024-010-HC.jpg','CFD-WF-022-007-HC.jpg','CFD-BF-037-022-N.jpg','CFD-BF-021-013-N.jpg','CFD-BF-047-003-N.jpg','CFD-WF-002_03.jpg','CFD-WF-112_03.jpg','CFD-WF-014_03.jpg','CFD-WF-083_08.jpg','CFD-BF-025_08.jpg','CFD-WF-122_08.jpg']
 
    #Randomly choose cb1 or cb2 to be study faces (first half of perps list) and the other half to be "suspects" in target absent lineups
    import random 
    ids = ids_fixed_cb1+ids_fixed_cb2
    cb = 'CB1'
    if random.randint(0, 1) == 1: 
        ids = ids_fixed_cb2 +ids_fixed_cb1 
        cb = 'CB2'
            
    num_lineups = len(ids)

    #Get identity numbers for perps list (unfortunately this code reorders the filenames so doesn't work well)
    filtered_rows = face_descriptions[face_descriptions['Filename'].isin(ids)]   # Filter rows in the DataFrame where 'Filename' is in the list 'ids'
    identity_nums_perps = filtered_rows['Identity'].tolist()   # Extract the 'Identity' values from the filtered rows
        
    #The first half of the list will become perps / target present trials. So let's just randomise the list so a slightly different group ends up as perps each time.
    import random
    random.shuffle(identity_nums_perps) # Shuffle the list in place

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
        
        #Get copy of first candidate perp, extract and record its index and assign it to perps dataframe
        fd_row_copy = fd_row.iloc[[0]].copy()
        fd_row_copy['Counterbalance'] = cb
        fd_row_copy['Index'] = fd_row_copy.index
        perps.append( fd_row_copy )
        
        #Mark selected face as used
        this_identity = perps[count].iloc[0]["Identity"]   #which identity was just selected Ppython subsetting is embarrassing inelegant!!!!)?
        face_descriptions.loc[face_descriptions["Identity"] == this_identity,"Used" ]=1 #set all rows with this identity to 1 (and nothing else!)
        
        
    perps = pd.concat(perps)    #Collapse the array of dataframes down into one big dataframe
       
    
    #paired = []
    car_conds = {
        1: "Veridical",
        2: "Caricature",
        3: "Anticaricature"
        }
    
    #Initialise dictionaries to hold dataframes for lineups
    # lineups = {key: None for key in range(0, num_lineups)}
    lineups = []
    
    #Now use a loop to create the lineup targets and suspects (first elements in each lineup dictionary key). This loop is a bit of a nightmare.
    for count, identity in enumerate(identity_nums_perps):  # Iterate through identity nums of randomly-sampled perps
        
        #which expression should this paired be?
        paired_expression = "Neutral"
        if (perps.iloc[count]["Expression"] == "Neutral"): 
            paired_expression = "Smiling" #If perp on this trial is neutral, switch to smiling 
        
        #Which car level should paired be?
        this_car_cond = car_conds[((count+1) - 1) % 3 + 1]  #Rotate through the car level condition of the paired face every three trials
        
        #Which type of trial is this? Target present or absent?
        lineup_type = 'Target present'
        facetype = 'Perp'
        if count >= num_lineups/2:
            lineup_type = 'Target absent'
            facetype = 'Innocent suspect'
        
        #First put perp / suspect test face in first row of lineup
        
        #find (should be unique) row in face_descriptions with same identity as perp, opposite expression and appropriate car level
        fd_row = []
        fd_row_copy = []
        fd_row = face_descriptions[
            (face_descriptions["Identity"] == identity) & 
            (face_descriptions["Expression"] == paired_expression) & 
            (face_descriptions["Test caricature"] == this_car_cond)
            ] 
        
        #assign a copy of first row to paired and add some extra info 
        fd_row_copy = fd_row.iloc[0:1].copy()
        fd_row_copy['Counterbalance'] = cb
        fd_row_copy['Index'] = fd_row_copy.index
        fd_row_copy['Face type'] = facetype
        fd_row_copy['Lineup type'] = lineup_type
        fd_row_copy['Lineup num'] = count
        
        #assign to this lineup key in lineups dict
        # lineups[count] = fd_row_copy
        lineups.append(fd_row_copy)
        
        #Now add the fillers to this lineup
        
        #This should get all candidate faces that have the same race, gender, opposite expression as perp, the appropriate car level and has not already been used
        
        #Match the race and gender of this new face to one of the perps
        this_race = perps.iloc[count]["Race"]
        this_gender = perps.iloc[count]["Gender"]
        
        fd_row = []
        fd_row_copy = []
        fd_row = face_descriptions[
            (face_descriptions["Used"] != 1) &
            (face_descriptions["Gender"] == this_gender) &
            (face_descriptions["Race"] == this_race) & 
            (face_descriptions["Expression"] == paired_expression) & 
            (face_descriptions["Test caricature"] == this_car_cond)
            ]
        
        #Randomise the order of the matching rows to shuffle filler candidates
        fd_row = fd_row.sample(frac=1)
        
        #Get the top (shuffled) filler candidates
        fd_row_copy = fd_row.iloc[0:num_fillers].copy()
        
        #Add the extra info needed
        fd_row_copy['Counterbalance'] = cb
        fd_row_copy['Index'] = fd_row_copy.index
        fd_row_copy['Face type'] = 'Filler'
        fd_row_copy['Lineup type'] = lineup_type
        
        #Assign to lineup dictionary
        # lineups[count] = pd.concat([lineups[count], fd_row_copy], ignore_index=False)
        fd_row_copy['Lineup num'] = count
        lineups.append(fd_row_copy)
        
        #Make sure these fillers now can't be sampled again on future iterations of this loop
        these_identities = fd_row_copy["Identity"]   
        face_descriptions.loc[face_descriptions['Identity'].isin(these_identities), 'Used'] = 1
        
    #Convert the list of daraframes to one big dataframe
    lineups = pd.concat(lineups)

#Now get rid of the second half of perps, which are just the suspects, so later you can find distances of test faces to just these perps in face space
    perps = perps.iloc[0:int(num_lineups/2)]
    
    #double check all perps/suspects got assigned four fillers before proceeding
    if lineups.shape[0] != num_lineups*(num_fillers+1):
        print('Lineup has too few fillers: ', lineups.shape[0])
    
    return perps, lineups
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
    Ematrix = squareform(pdist(np.asarray(dat),'euclidean'))
    # Ematrix = squareform(pdist(np.asarray(dat),'cityblock'))
    
    perps_distances = []    #empty list to accumulate all the pairs

    num_fake_subs = 50
    for fake_sub in range(num_fake_subs):
    
        #Don't forget that paired is now a dictionary where each key is associated with a dataframe whose rows are the faces in one lineup
        perps, paired = get_oldnew_trials(face_descriptions,gender)
        
        #summarise genders of output(for debugging)
        man_count = (paired['Gender'] == 'Man').sum()
        woman_count = (paired['Gender'] == 'Woman').sum()
        print(f"fake_subject: {fake_sub}, {man_count} 'Man' test faces, {woman_count} 'Woman' test faces")
        
        
        def find_min_dissimilarity_to_perps(lineup_df, perps, Ematrix):
            perps_indices = perps['Index'].values
            similarities = []

            for idx in lineup_df['Index']:
                # Retrieve the similarity values for the current face to all perps faces
                similarity_values = Ematrix[idx, perps_indices]
                # Find the min dissimilarity value
                min_dissimilarity = similarity_values.min()
                similarities.append(min_dissimilarity)
                
            # Add the maximum similarity values as a new column
            lineup_df['Nearest distance'] = similarities
            return lineup_df
        
        # Update each lineup in the 'paired' dataframe
        paired = find_min_dissimilarity_to_perps(paired, perps, Ematrix)

        paired['Simulated participant'] = fake_sub
        
        perps_distances.append(paired)
        
    perps_distances = pd.concat(perps_distances, ignore_index=True)
    
    distances_plot_data = perps_distances.groupby(['Test caricature','Face type', 'Lineup type','Simulated participant'])['Nearest distance'].mean().reset_index()



    ####THE DISTANCES PLOT!!!!!######
    fontsize = 22
    fig, axs = plt.subplots(1, 1, figsize=(6, 7))
    
    # colors = ["#4878CF", "#6ACC65", "#FFA500" ]
    colors = ['blue','red','green']
    customPalette = sns.set_palette(sns.color_palette(colors))
    
    hue_order = ["Anticaricature", "Veridical", "Caricature"]
    order = ['Perp', 'Innocent suspect']
    
    # Filter DataFrame for Target present rows and create the strip plot
    target_present = distances_plot_data[(distances_plot_data['Face type'] == 'Perp') | (distances_plot_data['Face type'] == 'Innocent suspect')]
    # target_present = distances_plot_data[distances_plot_data['Lineup type'] == 'Target present']
    
    #Create box plots with dodging
    sns.boxplot(ax=axs,data=target_present, x='Face type', y='Nearest distance', hue='Test caricature', order = order, hue_order = hue_order, dodge=True, boxprops=dict(alpha=.2),showfliers = False)
    
    sns.stripplot(ax=axs, x='Face type', y='Nearest distance', hue='Test caricature', data=target_present, order = order,
                  hue_order=hue_order, dodge=True, jitter=True)
    
    
    axs.tick_params(labelsize=fontsize)  # Increase tick label font size
    axs.legend(title='', fontsize=fontsize)  # Increase font size of legend title
    axs.set_xlabel('Test face', fontsize=fontsize)
    axs.set_ylabel('Distance to nearest perp face', fontsize=fontsize)
    plt.tight_layout()
    
    # Make the legend box background more transparent
    handles, labels = axs.get_legend_handles_labels()
    # legend = axs.legend(handles[0:3], labels[0:3], title='', fontsize=fontsize-2)
    legend = axs.legend(handles[0:3], labels[0:3], title='', fontsize=fontsize-4, loc='upper left', bbox_to_anchor=(0, 1), framealpha=0.3)
    legend.get_frame().set_alpha(0.3)
    
    plt.show()
    
    # # axs[0].set_title('Target present', fontsize=fontsize)
    # axs[0].tick_params(labelsize=fontsize)  # Increase tick label font size
    # axs[0].set_xlabel('')  # Suppress x-axis label
    # axs[0].legend(title='', fontsize=fontsize)  # Increase font size of legend title
    # # Set y-axis ticks every 200 starting at 400 and finishing at 1000
    # # axs[0].set_ylim(350, 1100)  # Set y-axis limits
    # # axs[0].set_yticks(np.arange(400, 1001, 200))

    
    # # target_absent = distances_plot_data[distances_plot_data['Lineup type'] == 'Target absent']
    # # sns.stripplot(ax=axs[1], x='Face type', y='Nearest distance', hue='Test caricature', data=target_absent,
    # #               hue_order=hue_order, dodge=True, jitter=True)
    # # axs[1].set_title('Target absent',fontsize=fontsize)
    # # axs[1].tick_params(labelsize=fontsize)  # Increase tick label font size
    # # axs[1].set_xlabel('')  # Suppress x-axis label
    # # axs[1].get_legend().remove()  # Suppress legend on the second plot
    # # # Set y-axis ticks every 200 starting at 400 and finishing at 1000
    # # # axs[1].set_ylim(350, 1100)  # Set y-axis limits
    # # # axs[1].set_yticks(np.arange(400, 1001, 200))

    
    # # Set x-axis label font size
    # for ax in axs:
    #     ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
    
    # # Set y-axis label font size
    # for ax in axs:
    #     ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)
        
    # # Adjust layout
    # plt.tight_layout()
        

    
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
def compute_and_plot_AUC(ROC_data):
    
    #I am revising this one to compute the AUCs at the group level only, so that I can efficiently do a partial interval based on the smallest asymptotic false alarm rate. This will be the case for lineups because false alarms can't exceed 1/num options but I can run this for matching and old/new
    
    # Filter the data by 'Face type' and 'Test caricature'
    conditions = ['Anticaricature', 'Veridical', 'Caricature']
    face_types = ['Perp', 'Innocent suspect']
    
        
    # Compute the maximum false alarm rates for each condition
    max_fas = {}
    for condition in conditions:
        max_fas[condition] = ROC_data[(ROC_data['Test caricature'] == condition) & (ROC_data['Face type'] == 'Innocent suspect')]['mean'].max()
    
    # Determine the integration limit
    limit = min(max_fas.values())

    partial_aucs = {}
    for condition in conditions:
        
        data = ROC_data[ROC_data['Test caricature'] == condition]
        
        fa_data = data[data['Face type'] == 'Innocent suspect']
        hit_data = data[data['Face type'] == 'Perp']

        # Ensure data are sorted by criterion
        fa_data = fa_data.sort_values(by='Criterion')
        hit_data = hit_data.sort_values(by='Criterion')
        
        # Truncate the false alarm data at the integration limit
        fa_truncated = fa_data[fa_data['mean'] <= limit]
        hit_truncated = hit_data[hit_data['Criterion'].isin(fa_truncated['Criterion'])]
    
        # Ensure both arrays have the same length
        fa_truncated = fa_truncated.head(len(hit_truncated))
        hit_truncated = hit_truncated.head(len(fa_truncated))
        
        # Interpolate missing points (optional, but recommended for sparse data)
        new_fas = np.linspace(0.001, limit, num=100)  # Common x-axis for interpolation
        
        from scipy.interpolate import interp1d
        
        f1 = interp1d(fa_truncated['mean'], hit_truncated['mean'], kind='linear', fill_value="extrapolate")
        hit_interp = f1(new_fas)
        
                
        # Compute the area using the trapezoidal rule
        partial_aucs[condition] = np.trapz(hit_interp, new_fas)
        
        

    # Create a bar plot of the AUC values
    fontsize = 22
    plt.figure(figsize=(8, 6))
    
    colors = ['blue','red','green']
    
    plt.bar(partial_aucs.keys(), partial_aucs.values(), color = colors, alpha = .3)

    plt.xlabel('Test caricature', fontsize=fontsize)
    plt.ylabel('Discriminability (Partial AUC)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    plt.show()
    
    return partial_aucs
    

        




# %%
##########################
def get_ROC_data(perps_distances):
    
    #Loop through criterion distances and count accuracy, hits and false alarms 
    
    criteria = np.linspace(
        np.min(perps_distances['Nearest distance']),
        np.max(perps_distances['Nearest distance']), 
                19)
    # criteria = np.append(criteria,np.Inf)    #Use a crazy high criterion value which will guarantee 100% hits and will cause the ROC to automatically extend all the way to the top right corner and simpify AUC computations.
    num_criteria = len(criteria)
    
    # Initialize empty lists to store data extracted for each criterion
    ROC_data_list = []
    ROC_data_ss_list = []
    
    # Define a function to apply to each lineup
    def assign_old_responses(group):
    
        min_distance = group['Nearest distance'].min()  # Find the minimum distance in the group
        
        group['Old responses'] = np.where((group['Nearest distance'] == min_distance) & (min_distance < criteria[i]), 1, 0)
        
        return group
    
    for i in range(num_criteria):
        
        #You want to build up new dataframes ROC_data and ROC_data_ss that derive from perps_distances, but expound its rows by criteria but at the same average over lineups per simulated participant and condition (ROC_data_ss, to be used to compute individual participant AUC) or over participants and lineups per conditions (ROC_data, to be used to plot ROC)
    
#####Get the old responses for this criterion level:
    
        # Create a temporary DataFrame perps_distances_temp that can be processed for each criterion level
        perps_distances_temp = perps_distances.copy()
        
        #Now prepare to process each lineup separately
        grouped = perps_distances_temp.groupby(['Lineup num','Simulated participant'])    # Group the DataFrame by the "Lineup num" column
        
        # Apply the function to each group and concatenate the results
        perps_distances_temp = grouped.apply(assign_old_responses).reset_index(drop=True)
        
        
   
        
   
######Now you have a new column in perps_distances_temp with old responses, now average it appropriately (You are still inside criteria loop)  

        ############
        # Create this criterion's portion of ROC_data_ss ...
        mean_distances = perps_distances_temp.groupby(['Simulated participant', 'Face type', 'Lineup type', 'Test caricature'])['Old responses'].mean().reset_index()
        
        # Add 'criteria' column to mean_ci DataFrame
        mean_distances['Criterion'] = criteria[i]
        
        #I like toi be organised. This will make ROC_data_ss easier to read when it's encountered later in the program
        mean_distances.sort_values(by=['Simulated participant', 'Lineup type', 'Test caricature', 'Face type'], inplace=True)
        
        # ... and assign it to list to be dataframed below
        ROC_data_ss_list.append(mean_distances)
        
        
        
        ###########
        #Create this criterion's portion of ROC_data ...
        mean_ci = mean_distances.groupby(['Face type', 'Lineup type', 'Test caricature']).agg({'Old responses': ['mean', 'sem']}) # Compute mean and confidence interval collapsing over 'Simulated_Participant'
        mean_ci.columns = mean_ci.columns.droplevel(0)  # Remove the 'Old responses' label from the column index
        
        # Calculate 95% confidence interval (using a multiplier of 1.96)
        mean_ci['CI'] = 1.96 * mean_ci['sem']
        
        # Add 'criteria' column to mean_ci DataFrame
        mean_ci['Criterion'] = criteria[i]

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
      
 
     fig, axs = plt.subplots(1, 1, figsize=(7, 7))

     # colors = ["#4878CF", "#6ACC65", "#FFA500" ]
     colors = ['blue','red','green']
     customPalette = sns.set_palette(sns.color_palette(colors))
    
     ordered_labels = ["Anticaricature", "Veridical", "Caricature"]
     # plt.gca().set_prop_cycle(color=colors)
    
     lines = []
     # for level in caricature_levels:
     for level, color in zip(ordered_labels,colors):

         # Filter data for the current caricature level
         data_level = ROC_data[ROC_data['Test caricature'] == level]
        
         # Separate data for 'Hits' and 'False alarms'
         match_data = data_level[data_level['Face type'] == 'Perp']
         mismatch_data = data_level[data_level['Face type'] == 'Innocent suspect']
                
         # Plot ROC curve
         lines.append(axs.plot(mismatch_data['mean'], match_data['mean'], label=level, color = color, marker='o', markerfacecolor = color, markersize = 10)[0])
        
     # #     # Add shaded error area
     #     axs.fill_between(mismatch_data['mean'], match_data['mean'] - match_data['CI'], match_data['mean'] + match_data['CI'], alpha=0.2)
        
     # Draw a diagonal line where x=y
     axs.plot([0, 1], [0, 1], color='black', linewidth=2, label='Chance', linestyle='--')
    
     fontsize = 22
     # axs[0].set_title('Target present', fontsize=fontsize)
     axs.tick_params(labelsize=fontsize)  # Increase tick label font size
     axs.set_xlabel('Cumulative false alarm rate')  # Suppress x-axis label
     axs.set_ylabel('Cumulative hit rate')  # Suppress x-axis label
     axs.legend(title='', fontsize=fontsize)  # Increase font size of legend title

     # Set ticks every 0.2 units
     axs.set_xticks(np.arange(0, 1.1, 0.05))
     axs.set_yticks(np.arange(0, 1.1, 0.2))
     
     axs.set_ylim(-0.01, 1.01)  # Set y-axis limits
     # axs.set_xlim(-0.01, 1.01)  # Set x-axis limits
     axs.set_xlim(-.01, 0.21)  # Set x-axis limits
    
     axs.set_xlabel(axs.get_xlabel(), fontsize=fontsize)
     axs.set_ylabel(axs.get_ylabel(), fontsize=fontsize)
     
     # # Force the axis to be square
     # axs.set_aspect('equal', 'box')
    
     axs.grid(False)
        
     # Adjust layout
     plt.tight_layout()
     
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


#########Proprocess images and project them into model space
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
Ematrix, perps_distances = get_trial_distances(face_descriptions, dat, gender = "None")

#plot_distances_by_car_level(Ematrix, face_descriptions)

#Compute cumulative hits and false alarms
ROC_data_ss, ROC_data = get_ROC_data(perps_distances)

plot_ROCs(ROC_data)

AUC_data = compute_and_plot_AUC(ROC_data)

    

print('audi5000')
