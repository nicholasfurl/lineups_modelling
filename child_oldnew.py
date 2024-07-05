
#README: child_oldnew.py expands on the framework started with RQ1_oldnew. Except it works with the child faces and manipulates caricature and both study and test






# %%
###################
# Function to extract filename and split by hyphen
def extract_and_split(path,delimiter_string):
    filename = os.path.basename(path)  # Extract filename from path
    filename_without_extension = os.path.splitext(filename)[0]  # Remove extension if needed
    parts = filename_without_extension.split(delimiter_string)  # Split filename by hyphen

    return parts
########################





# %%
#Uses sorted_pairs - zipped up filenames, expressions and genders, sorted by expression (created by one of the get*data functions)
#returns a dataframe new_rows that can be concatenated to face_descriptions
###################
def process_filename_data(sorted_pairs, folder_car, id_it):
    
    #Make list of filenames
    filenames = [pair[0] for pair in sorted_pairs] # Extract the original filenames from the sorted pairs
     
    #Make list containg the (formatted) expression
    expression_strings = [pair[1] for pair in sorted_pairs] #extract sorted expression strings
    expressions = ["Neutral" if parts.lower().startswith("n") else "Happy" for parts in expression_strings]
    
    #Make list containing the (formatted) gender
    gender_strings = [pair[2] for pair in sorted_pairs] #extract sorted raw gender strings (Ms and Fs)
    genders = ["Man" if parts.startswith("M") else "Woman" for parts in gender_strings]
   
    #Make list containing the caricature level
    temp = [folder_car for _ in sorted_pairs]
    #Format this list
    caricatures = []
    for s in temp:
        modified_string = s.lstrip("\\").replace(" ", "").capitalize()   # Remove leading "\", capitalize the next character, and remove spaces
        caricatures.append(modified_string)

    #Label identities. Strongly assumes filenames sorted by expression, which equal number of identities for each expression
    num_happy = expressions.count("Happy")  # Count the number of instances of "Happy"  
    identities = []
    for i in range(2):
        identities.extend(range(id_it + 1, id_it + num_happy + 1))  # Generate identities
     
    new_rows = {
           "Filename": filenames, 
           "Caricature level": caricatures, 
           "Gender": genders, 
           "Expression": expressions, 
           "Identity": identities
     } 
    
    new_rows = pd.DataFrame(new_rows)
    
    return new_rows
###############################









# %%
#returns a dataframe new_rows that can be concatenated to face_descriptions
###################
def get_deffs_data(path_cars):
    
    #get DEFFS files
    filenames_deffs = sorted(glob.glob(path_cars+r'\DEFFS\*.jpg'))
    # filenames_deffs = sorted(glob.glob(base_path+folders_car[0]+folders_familiarity[0]+folders_imagesets[1]+'\*.jpg'))

    #break down info in filenames
    split_filenames = [extract_and_split(path,'_') for path in filenames_deffs]
        
    #Make sure filenames and their splits are sorted by expression
    zipped = zip(filenames_deffs, [x[2] for x in split_filenames], [x[1] for x in split_filenames])    # Zip the original filenames with the corresponding second chunks from split_filenames
    sorted_pairs = sorted(zipped, key=lambda x: x[1])   # Sort the zipped pairs based on the second chunk in split_filenames

    return sorted_pairs
###############################











# %%
#returns a zip, with filenames, then expressions, then genders
###################
def get_cafe_data(path_cars):
    
    #get CAFE files
    filenames_cafe = sorted(glob.glob(path_cars+r'\CAFE\*.jpg'))
    # filenames_cafe = sorted(glob.glob(base_path+folders_car[0]+folders_familiarity[0]+folders_imagesets[0]+'\*.jpg'))

    #break down info in filenames
    split_filenames = [extract_and_split(path,'-') for path in filenames_cafe]
        
    #Sorted pairs contains filenames, then expressions, then genders, sorted by expression
    zipped = zip(filenames_cafe, [x[1] for x in split_filenames], [x[2] for x in split_filenames])    # Zip the original filenames with the corresponding second chunks from split_filenames
    sorted_pairs = sorted(zipped, key=lambda x: x[1])   # Sort the zipped pairs based on the second chunk in split_filenames
    
    return sorted_pairs
###############################




  




# %%
#returns a zip, with filenames, then expressions, then genders
###################
def get_nihm_data(path_cars):
    
    #get nihm files
    filenames_nihm = sorted(glob.glob(path_cars+r'\NIHM\*.jpg'))
    # filenames_nihm = sorted(glob.glob(base_path+folders_car[0]+folders_familiarity[0]+folders_imagesets[2]+'\*.jpg'))

    #break down info in filenames
    split_filenames = [extract_and_split(path,'-') for path in filenames_nihm]
        
    #Sorted pairs contains filenames, then expressions, then genders, sorted by expression
    zipped = zip(filenames_nihm, [x[0][3] for x in split_filenames], [x[0][0] for x in split_filenames])    # Zip the original filenames with the corresponding second chunks from split_filenames
    sorted_pairs = sorted(zipped, key=lambda x: x[1])   # Sort the zipped pairs based on the second chunk in split_filenames
    
    return sorted_pairs
###############################









# %%
#returns face_descriptions dataframe
###################
def get_all_files():
    
    base_path = r'C:\matlab_files\lineups_modelling\Child stimuli'

    #In the base path, there are separate folders for caricature levels
    folders_car = [r'\anti caricatured', r'\caricatured', r'\veridical']
    
    #for now, we'll just do unfamiliar, like the empirical study
    #folders_familiarity = ['famous', 'unfamiliar']
    folders_familiarity = [r'\unfamiliar']

    # #three unfamiliar image sets
    # folders_imagesets = [r'\CAFE',r'\DEFFS',r'\NIHM']

    #Initialise dataframe to hold image data
    face_descriptions = {
           "Filename": [], 
           "Caricature level": [], 
           "Gender": [], 
           "Expression": [], 
           "Identity": []
     }
    face_descriptions = pd.DataFrame(face_descriptions)

    #Loop through caricature condition folders to extract files
    for folder_car in folders_car:
        
        path_car = base_path+folder_car+folders_familiarity[0]
        
        #start id count over, as 
        id_it = 0     #To name identities
        
        #get and process cafe data
        sorted_pairs = get_cafe_data(path_car)
        face_descriptions = pd.concat(
            [face_descriptions,
             process_filename_data(sorted_pairs, folder_car, id_it)
             ]
            )
        
        id_it = int(face_descriptions['Identity'].iloc[-1])  #update the current identity to be the last one assigned
        
        #get and process deffs data
        sorted_pairs = get_deffs_data(path_car)
        face_descriptions = pd.concat(
            [face_descriptions,
             process_filename_data(sorted_pairs, folder_car, id_it)
             ]
            )
        
        id_it = int(face_descriptions['Identity'].iloc[-1])  #update the current identity to be the last one assigned
        
        #get and process nihm data
        sorted_pairs = get_nihm_data(path_car)
        face_descriptions = pd.concat(
            [face_descriptions,
             process_filename_data(sorted_pairs, folder_car, id_it)
             ]
            )
        
    face_descriptions = face_descriptions.reset_index()
    
    return face_descriptions
###############################








# %%
###############################
def get_model_space(face_descriptions):
    
    # #Set up vgg16 model
    # from keras.applications import vgg16
    # from keras.models import Model
    
    # model = vgg16.VGG16(weights='imagenet', include_top=True)
    # model2 = Model(model.input, model.layers[-2].output)
    
    from keras_vggface.vggface import VGGFace
     
    # # Convolution Features
    # model2 = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    
    # Convolution Features
    model2 = VGGFace(model='senet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    
    
    #########Proprocess images and project them into vgg16 space
    import keras.utils as image
    #from keras.applications.vgg16 import preprocess_input
    from keras_vggface.utils import preprocess_input
    from keras.models import Model
    
    dat = []
    imgs = []
    for count, imgf in enumerate(face_descriptions['Filename']):
        
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
        
    return dat
###############################

    
    



# %%
#######################
# Visualise category exemplars in flattened 2D (t-SNE and PCA) spaces 
def tsne_and_pca(dat,face_descriptions):
   
    #t-SNE
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42,perplexity=len(dat)-1)  #Perplexity must be less than number of samples
    flat_space_tsne = tsne.fit_transform(np.asarray(dat))    #samples (i.e. pictures) by dimensions (2)
    visualise_flat_space(flat_space_tsne, face_descriptions,'TSNE')

    #PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    flat_space_pca = pca.fit_transform(np.asarray(dat))
    visualise_flat_space(flat_space_pca, face_descriptions,'PCA')
###############################





# %%
##########################
def visualise_flat_space(flat_space,face_descriptions,plot_label):
    
    colors = ["#4878CF", "#FFA500", "#6ACC65", ]
    customPalette = sns.set_palette(sns.color_palette(colors))
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))  # 1 row, 3 columns
    
    # Modify font sizes using rcParams
    graph_font = 14
    plt.rcParams.update({
        'font.size': graph_font,          # Base font size
        'axes.labelsize': graph_font,     # Size of axis labels
        'axes.titlesize': graph_font,     # Size of plot title
        'legend.fontsize': graph_font,    # Size of legend text
    })
         
    #plot genders
    # plt.figure()
    sns.scatterplot(x=flat_space[:, 0], y=flat_space[:, 1], hue=face_descriptions['Gender'], ax=axes[0,0])
    
    #plot caricature levels    
    # custom_palette = {'Anticaricatured': 'blue', 'Veridical': 'orange', 'Caricatured': 'green'} # Define custom color palette
    # plt.figure()
    sns.scatterplot(x=flat_space[:, 0], y=flat_space[:, 1], hue=face_descriptions['Caricature level'], ax=axes[0,1])
    
    #plot expressions
    # plt.figure()
    sns.scatterplot(x=flat_space[:, 0], y=flat_space[:, 1], hue=face_descriptions['Expression'], ax=axes[1,0])
    
    #plot identities
    # plt.figure()
    sns.scatterplot(x=flat_space[:, 0], y=flat_space[:, 1], hue=face_descriptions['Identity'].astype('category'), ax=axes[1,1])
    
    #the legend in the identity plot will have 65 entries and isn't viewable so remove
    axes[1,1].get_legend().remove()
    
    # Add common x and y axis labels
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlabel(plot_label+" dimension 1")
            ax.set_ylabel(plot_label+" dimension 2")
##############################
  
  
  
  


# %%
#######################################
def plot_distances_by_car_level(Ematrix, face_descriptions):
    
    #plot distances between caricatures, veridical and anticaricatures
    # Get the indices corresponding to each combination of Caricature level and Expression
    indices = {
        ('Neutral', 'Caricatured'): face_descriptions[(face_descriptions['Expression'] == 'Neutral') & (face_descriptions['Caricature level'] == 'Caricatured')].index,
        ('happy', 'Caricatured'): face_descriptions[(face_descriptions['Expression'] == 'Happy') & (face_descriptions['Caricature level'] == 'Caricatured')].index,
        ('Neutral', 'Veridical'): face_descriptions[(face_descriptions['Expression'] == 'Neutral') & (face_descriptions['Caricature level'] == 'Veridical')].index,
        ('Happy', 'Veridical'): face_descriptions[(face_descriptions['Expression'] == 'Happy') & (face_descriptions['Caricature level'] == 'Veridical')].index,
        ('Neutral', 'Anticaricatured'): face_descriptions[(face_descriptions['Expression'] == 'Neutral') & (face_descriptions['Caricature level'] == 'Anticaricatured')].index,
        ('Happy', 'Anticaricatured'): face_descriptions[(face_descriptions['Expression'] == 'Happy') & (face_descriptions['Caricature level'] == 'Anticaricatured')].index
    }
    
    Ematrix_temp = Ematrix.copy()
    Ematrix_temp[np.triu_indices_from(Ematrix_temp)] = np.nan    # Set upper triangle and diagonal to NaN
    
    # Calculate distances for each combination
    distances = {}
    for key, idx in indices.items():
        distances[key] = Ematrix_temp[np.ix_(idx, idx)]
       
    # Create a new DataFrame
    data = {'Distance': [], 'Expression': [], 'Caricature level': []}
    for key, dist in distances.items():
        data['Distance'].extend(dist.flatten())
        data['Expression'].extend([key[0]] * len(dist.flatten()))
        data['Caricature level'].extend([key[1]] * len(dist.flatten()))
    
    df = pd.DataFrame(data)
        
        # Plot splitplot superimposed over boxplots
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ["#4878CF",  "#6ACC65", "#FFA500"]
    customPalette = sns.set_palette(sns.color_palette(colors))
    
    # Modify font sizes using rcParams
    plt.rcParams.update({
        'font.size': 18,          # Base font size
        'axes.labelsize': 18,     # Size of axis labels
        'axes.titlesize': 18,     # Size of plot title
        'legend.fontsize': 18,    # Size of legend text
    })
    
    caricature_order = ['Anticaricatured','Veridical','Caricatured' ]
    
    # Create boxplots
    sns.boxplot(data=df, x='Caricature level', y='Distance', dodge=True, ax=ax, linewidth=1,order=caricature_order, showfliers=False, palette = customPalette)
    
    # Create splitplot
    sns.stripplot(data=df, x='Caricature level', y='Distance',  dodge=True, jitter=True, ax=ax, color='black', alpha=0.1, order=caricature_order)
    
    # Customize plot
    ax.set_ylabel('Distance')
    # ax.set_title('Euclidean Distance by Caricature level and Expression')
    
    plt.tight_layout()
    plt.show()
#######################################








# %%
##############
def get_oldnew_trials(face_descriptions,gender):
    
    num_study_faces = 16  #number of study / old faces, must be 18 if I'm to balance gender, expression, study and test caricature conditions
       
    # Randomly sample the 8 study and 8 test male identities
    identity_nums_male = face_descriptions[face_descriptions['Gender'] == 'Man']['Identity'].unique().astype(int)   #should be 24 total
    identity_nums_male_used = np.random.choice(identity_nums_male,size=num_study_faces,replace=False)
      
    # Randomly sample the 8 study and 8 test male identities
    identity_nums_female = face_descriptions[face_descriptions['Gender'] == 'Woman']['Identity'].unique().astype(int)   #should be 41 total
    identity_nums_female_used = np.random.choice(identity_nums_female,size=num_study_faces,replace=False)
    
    #split into old and new
    half_size = len(identity_nums_male_used) // 2
    identity_nums_male_used_old = identity_nums_male_used[:half_size]   #eight faces
    identity_nums_male_used_new = identity_nums_male_used[half_size:2*half_size] #eight faces
    identity_nums_female_used_old = identity_nums_female_used[:half_size] #eight faces
    identity_nums_female_used_new = identity_nums_female_used[half_size:2*half_size] #eight faces
    
    #Assignments I want to make to old and new male and female lists
    #study expression, study caricature, test expression, test caricature
    condition_list = []
    condition_list.append(["Happy","Caricatured","Neutral","Caricatured"])
    condition_list.append(["Happy","Anticaricatured","Neutral","Anticaricatured"])
    condition_list.append(["Happy","Caricatured","Neutral","Anticaricatured"])
    condition_list.append(["Happy","Anticaricatured","Neutral","Caricatured"])
    condition_list.append(["Neutral","Caricatured","Happy","Caricatured"])
    condition_list.append(["Neutral","Anticaricatured","Happy","Anticaricatured"])
    condition_list.append(["Neutral","Caricatured","Happy","Anticaricatured"])
    condition_list.append(["Neutral","Anticaricatured","Happy","Caricatured"])
    
    study_stimuli_men = []
    test_stimuli_old_men = []
    test_stimuli_new_men = []
    study_stimuli_women = []
    test_stimuli_old_women = []
    test_stimuli_new_women = []
    for count, condition in enumerate(condition_list):
        
        #study males (uses old list but first two entries in condition)
        study_stimuli_men.append(
            face_descriptions[
                (face_descriptions["Identity"] == identity_nums_male_used_old[count]) & 
                (face_descriptions["Expression"] == condition[0]) & 
                (face_descriptions["Caricature level"] == condition[1])]
            )
        
        #study females (uses old list but first two entries in condition)
        study_stimuli_women.append(
            face_descriptions[
                (face_descriptions["Identity"] == identity_nums_female_used_old[count]) & 
                (face_descriptions["Expression"] == condition[0]) & 
                (face_descriptions["Caricature level"] == condition[1])]
            )
        
        #test males old (uses old list but second two entries in condition)
        test_stimuli_old_men.append(
            face_descriptions[
                (face_descriptions["Identity"] == identity_nums_male_used_old[count]) & 
                (face_descriptions["Expression"] == condition[2]) & 
                (face_descriptions["Caricature level"] == condition[3])]
            )
        
        #test females old (uses old list but second two entries in condition)
        test_stimuli_old_women.append(
            face_descriptions[
                (face_descriptions["Identity"] == identity_nums_female_used_old[count]) & 
                (face_descriptions["Expression"] == condition[2]) & 
                (face_descriptions["Caricature level"] == condition[3])]
            )
        
        #new males (uses new list and second two entries in condition)
        test_stimuli_new_men.append(
            face_descriptions[
                (face_descriptions["Identity"] == identity_nums_male_used_new[count]) & 
                (face_descriptions["Expression"] == condition[2]) & 
                (face_descriptions["Caricature level"] == condition[3])]
            )
        
        #new females (uses new list and second two entries in condition)
        test_stimuli_new_women.append(
            face_descriptions[
                (face_descriptions["Identity"] == identity_nums_female_used_new[count]) & 
                (face_descriptions["Expression"] == condition[2]) & 
                (face_descriptions["Caricature level"] == condition[3])]
            )
        
    if gender == "Man":
        
        study_stimuli = study_stimuli_men
        test_stimuli_old = test_stimuli_old_men
        test_stimuli_new = test_stimuli_new_men
        
    elif gender == "Woman":
        
        study_stimuli = study_stimuli_women
        test_stimuli_old = test_stimuli_old_women
        test_stimuli_new = test_stimuli_new_women
        
    else:
           
        #concatenate lists of series
        study_stimuli = study_stimuli_men+study_stimuli_women
        test_stimuli_old = test_stimuli_old_men+test_stimuli_old_women
        test_stimuli_new = test_stimuli_new_men+test_stimuli_new_women 
    
    #collapse lists of series into single dataframe
    study_stimuli = pd.concat(study_stimuli)
    test_stimuli_old = pd.concat(test_stimuli_old)
    test_stimuli_new = pd.concat(test_stimuli_new)
    
    return study_stimuli, test_stimuli_old, test_stimuli_new
######################################






# %%
##############
def get_test_distances(Ematrix, face_descriptions, gender, num_fake_subs):

    #Initialise dataframe to hold distances results
    test_distances = {
        'Simulated participant': [],
        'Nearest distance': [],
        'Old': [],
        'Study caricature': [],
        'Test caricature': [],
        'Gender': []
    }
    test_distances = pd.DataFrame(test_distances)
     
    for fake_sub in range(num_fake_subs):
    
        #assemble stimuli to be used in study and test sets
        study_stimuli, test_stimuli_old, test_stimuli_new = get_oldnew_trials(face_descriptions,gender)
        
        #For purposes of iterating in a single loop
        test_stimuli = pd.concat((test_stimuli_old,test_stimuli_new))
        
        #So I can track if accessing old or new test faces during the looping below
        num_old_tests = test_stimuli_old.shape[0]   #should be 16, 8 males, 8 females, if Gender == "All"
        
        # Iterate over all (old and new) test faces and get nearest distance to study face for each
        it = 0
        for idx, row in test_stimuli.iterrows():
                
            # Get the distances to all items in 'perps' using the row/column index
            distances = Ematrix[idx, study_stimuli.index]   #There's an accidental column now called "index" but we want the "real" index
            min_distance = np.min(distances)
               
            #sort out variables dependent on whether a face was studied         
            old = "New"
            study_car = "New"
            study_expression = np.nan
            if it < num_old_tests:
                study_expression = study_stimuli['Expression'].iloc[it]
                study_car = study_stimuli['Caricature level'].iloc[it]
                old = "Old"
            
            new_row_data = {
                'Simulated participant': fake_sub,
                'Nearest distance': min_distance,
                'Old': old,
                'Study caricature': study_car,
                'Test caricature': row['Caricature level'],
                'Gender': row['Gender'],
                'Study expression': study_expression,
                'Test expression': row['Expression'],
            }
        
            test_distances = pd.concat([test_distances,pd.DataFrame([new_row_data])], ignore_index=True)
            
            it = it+1
    
    return test_distances
#######################################









# %%
#################################
def plot_distances_by_study_conditions(test_distances):
    
    plt.figure(figsize=(6, 6)) 
    
    # Modify font sizes using rcParams
    plt.rcParams.update({
        'font.size': 14,          # Base font size
        'axes.labelsize': 14,     # Size of axis labels
        'axes.titlesize': 14,     # Size of plot title
        'legend.fontsize': 12,    # Size of legend text
    })
     
    colors = ["#4878CF", "#FFA500", "#6ACC65"]
    customPalette = sns.set_palette(sns.color_palette(colors))
    
    order = ["Anticaricatured", "Caricatured", "New"]
   
    sns.stripplot(data=test_distances, y = "Nearest distance", x='Study caricature', hue='Test caricature', jitter=True, dodge=True, order = order, hue_order = order)
    #alpha = .4
    
    sns.boxplot(data=test_distances, y = "Nearest distance", x='Study caricature', hue='Test caricature', dodge=True, order = order, hue_order = order, showfliers=False,boxprops=dict(alpha=.3))

    handles, labels = plt.gca().get_legend_handles_labels()   # Get handles and labels for legend
    
    plt.legend(handles=handles[3:5], labels=labels[3:5], title = "Test caricature", loc='upper center',framealpha=0.5)
    #, bbox_to_anchor=(1, 0.5))
#########################################







# %%
##########################
def get_ROC_data(test_distances):
    
    #Loop through criterion distances and count accuracy, hits and false alarms 
    num_criteria = 20
    criteria = np.linspace(
        np.min(test_distances['Nearest distance']),
        np.max(test_distances['Nearest distance']), 
               num_criteria)    

    # Initialize an empty lists to store data extracted for each criterion
    ROC_data_list = []
    ROC_data_ss_list = []

    # Create a temporary DataFrame perps_distances_temp for each iteration
    perps_distances_temp = test_distances.drop(columns = ['Old','Gender','Study expression','Test expression']).copy()

    for i in range(num_criteria):

        # Add a new column with 1 where Distance is less than the current criteria value, and 0 otherwise
        perps_distances_temp["Old responses"] = (perps_distances_temp['Nearest distance'] < criteria[i]).astype(int)

        # Compute mean for every 'Simulated_Participant', 'Caricature level', and 'Match' group
        mean_distances = perps_distances_temp.groupby(['Simulated participant', 'Study caricature', 'Test caricature']).mean()
        
        # Compute mean and confidence interval collapsing over 'Simulated_Participant'
        mean_ci = mean_distances.groupby(['Study caricature', 'Test caricature']).agg({'Old responses': ['mean', 'sem']})
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
#############################################









# %%
############################
def plot_ROCs(ROC_data):
    
    # Plot ROC curves for different caricature levels
    plt.figure(figsize=(8, 8))  # Adjust width and height as needed
        
    # Modify font sizes using rcParams
    plt.rcParams.update({
        'font.size': 14,          # Base font size
        'axes.labelsize': 14,     # Size of axis labels
        'axes.titlesize': 14,     # Size of plot title
        'legend.fontsize': 12,    # Size of legend text
    })

    caricature_levels = ['Anticaricatured', 'Caricatured'] 

    colors = ['blue', 'blue', 'orange', 'orange']
    lines = ['--', '-', '--', '-']
    it = 0
        
    for test_level in caricature_levels:
        
        #get false alarms for this test level, to be used for both study conditions
        fas = ROC_data[(ROC_data['Study caricature'] == 'New') & (ROC_data['Test caricature'] == test_level)]
        
        for study_level in caricature_levels:
            
            #appropriate hit rate
            hits = ROC_data[(ROC_data['Study caricature'] == study_level) & (ROC_data['Test caricature'] == test_level)]
                    
        #     # Plot ROC curve
            plt.plot(fas['mean'], hits['mean'], label='Study '+study_level+', Test '+test_level, color = colors[it], linestyle = lines[it], marker='o', markersize=5)
            
        #     # Add shaded error area
            plt.fill_between(fas['mean'], hits['mean'] - hits['CI'], hits['mean'] + hits['CI'], alpha=0.3, color = colors[it], linestyle = lines[it])
            
            it = it+1
            
    # Draw a diagonal line where x=y
    plt.plot([0, 1], [0, 1], color='grey', linestyle='-', label = "Chance")

    # Add labels and legend
    plt.xlabel('False alarm rate')
    plt.ylabel('Hit rate')
    plt.legend(title='Caricature Level')
#######################################  








# %%
############################
def compute_and_plot_AUC(ROC_data_ss):
    
    auc_diag = np.trapz([0, 1],[0, 1])
    participants = ROC_data_ss['Simulated participant'].unique()
    car_levels = ['Anticaricatured', 'Caricatured'] 

    #Initialise dataframe to hold auc results
    AUC = {
        'Simulated participant': [],
        'Study caricature': [],
        'Test caricature': [],
        'Area under the curve (AUC)': []
    }
    AUC = pd.DataFrame(AUC)
      
    for participant in participants:
        
        for test_level in car_levels:
            
            this_fas_data = ROC_data_ss[
                (ROC_data_ss['Simulated participant'] == participant) & 
                (ROC_data_ss['Study caricature'] == "New") &
                (ROC_data_ss['Test caricature'] == test_level)
                ]["Old responses"]
            
            for study_level in car_levels:
        
                this_hits_data = ROC_data_ss[
                    (ROC_data_ss['Simulated participant'] == participant) & 
                    (ROC_data_ss['Study caricature'] == study_level) &
                    (ROC_data_ss['Test caricature'] == test_level)
                    ]["Old responses"]
                
                new_row_data = {
                    'Simulated participant': participant,
                    'Study caricature': study_level,
                    'Test caricature': test_level,
                    'Area under the curve (AUC)': np.trapz(this_hits_data,this_fas_data) - auc_diag   #careful, the argument for y comes first here!
                }
                
                AUC = pd.concat([AUC,pd.DataFrame([new_row_data])], ignore_index=True)
                
                
    plt.figure(figsize=(6, 6)) 
    
    # Modify font sizes using rcParams
    plt.rcParams.update({
        'font.size': 14,          # Base font size
        'axes.labelsize': 14,     # Size of axis labels
        'axes.titlesize': 14,     # Size of plot title
        'legend.fontsize': 10,    # Size of legend text
    })
     
    colors = ["#4878CF", "#FFA500", "#6ACC65"]
    customPalette = sns.set_palette(sns.color_palette(colors))

    order = ["Anticaricatured", "Caricatured"]

    sns.stripplot(data=AUC, y = "Area under the curve (AUC)", x='Study caricature', hue='Test caricature', jitter=True, dodge=True, order = order, hue_order = order)
    #, alpha = .4

    sns.boxplot(data=AUC, y = "Area under the curve (AUC)", x='Study caricature', hue='Test caricature', dodge=True, order = order, hue_order = order, showfliers=False, boxprops=dict(alpha=.3))

    handles, labels = plt.gca().get_legend_handles_labels()   # Get handles and labels for legend

    plt.legend(handles=handles[2:4], labels=labels[2:4], title = "Test caricature", loc='lower left',framealpha=0.5)             

    return AUC
##################################
    
    
    
    

  
  
  

# %%
######################  
##MAIN BODY OF CODE
######################

import pandas as pd
import numpy as np
import glob 
import os
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)

print('running child_oldnew.py')

#Make dataframe with everything we need
face_descriptions = get_all_files()

#Get positions in model's high level face space
#dat is a list of np arrays, each (4096,) 
dat = get_model_space(face_descriptions)

#Visualise category exemplars in flattened 2D (t-SNE and PCA) spaces 
tsne_and_pca(dat,face_descriptions)  

#rows and cols of Ematrix align with rows of face_descriptions and dat
from scipy.spatial.distance import pdist, squareform
Ematrix = squareform(pdist(np.asarray(dat),'euclidean'))    #Get Euclidean distance matrix: 

#Plot distances between images at each separate caricature level so we can verify that distances are longer with greater caricature level
plot_distances_by_car_level(Ematrix, face_descriptions)

#Assemble stimuli and trials for simulated participants
#Third parameter controls genders used: arguments can be "All", "Man" or "Woman"
#Fourth parameter controls number of simulated participants: N = 399 in behavioural study
gender = 'All'
num_fake_subs = 399
test_distances = get_test_distances(Ematrix, face_descriptions, gender, num_fake_subs)

#Creates plot of distances to nearest neighbour for all test faces as a function of study and test caricature conditions
plot_distances_by_study_conditions(test_distances)

#Computes cumulative hits and false alarms for Study and Test caricature conditions
ROC_data_ss, ROC_data = get_ROC_data(test_distances)

plot_ROCs(ROC_data)

AUC_data = compute_and_plot_AUC(ROC_data_ss)




















print('audi5000')




        