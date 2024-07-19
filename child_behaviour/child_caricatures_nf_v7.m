%v7 1st May 2024 - continued improving plots to match neural network plots.
% ... Put Study car on y axis with test car dodged and put FAs on same plot as hits. 
%v7 4 April 2024 - found and fixed some bugs
%v6: Makes figures more publication-worthy
%v5 adds an analysis of group by quantile
%v4: allows more than one cfmt+ group at a time;
%v3: adds plot of proportion correct at different confidence levels


clear all;


addpath(genpath('C:\matlab_files\plotSpread'));

%changeable, orients the order of plots in figures
groups_in_cols = 1;
%default (group_setting == 0)is to organise plots by expertise group, with first plot experts
%and second non-experts. But if we flip group_setting to 1, then plots orgamised by
%CFMT+ median split with 0 low CFMT+ and 1 high CFMT+.
group_setting = 1;

%data cols: (1) event index, (2) participant id, (3) counterbalance group, (4) face response (0=old, 1=new), (5) confidence (0:10:100),
%(6) display, (7) test distinctiveness (0=caricature, 1=anticaricature), equal to column 11 (8) correct face response (0=old, 1=new),
%(9) stimulus sex (0=male, 1=female), (10) condition (0=car-car, 1=car-anticar, 2=anticar,car, 3=anticar-anticar),
%(11) study condition (0=caricature, 1=anticaricature), (12) test condition
%(0=caricature, 1=anticaricature), equal to column 7 (13) expertise group (0=experience, 1=no
%experience), (14) cfmt+ score.

%v3 is only different from v2 because I sorted the Excel by participant
%then by event index so it's easier to check the data columns against the
%quantile / index labels.

[big_data b c] = xlsread('C:\matlab_files\rif_2020\child_car_data\data_child_newold_task_copia_v3.xlsx');
%But then in the code below we assign big_data to data and then add to data 15 (is
%hit or not) and 16 (is FA or not)

%I want to double check something. There seem to be arbitary condition
%labels in column 10 for new items that should refer only to hits. I want
%to relabel to ensure there are nans for new items
for i=1:size(big_data,1);
    
    if big_data(i,8) == 1;   %if a new item
        
        big_data(i,10) = NaN;   %then it does not have one of the four condition labels
        big_data(i,11) = NaN;
        
    end;    %new item?
    
end;    %all trials loop

%for purposes of plotting
split_values = [-Inf prctile(big_data(:,14), 5) prctile(big_data(:,14), 95) Inf]; %list of cfmt+ boundaries to split data, the number of groups will be defined as numel(split_values)-1
% split_values = [-Inf prctile(big_data(:,14), 5) prctile(big_data(:,14), 95) Inf]; %list of cfmt+ boundaries to split data, the number of groups will be defined as numel(split_values)-1
cfmt_median_split = zeros(size(big_data,1),1);
%make new variable to hold cfmt+ group labelsusing a not at all elegent method that
%doesn't require any thinking
for i=1:size(big_data,1);   %loop through each row of the data file
    %for each row, loop through splits to check each
    for j=1:numel(split_values) - 1;   %loop through lower bounds (don't need final upper bound
        if big_data(i,14) >= split_values(j) & big_data(i,14) < split_values(j+1);
            cfmt_median_split(i,1) = j-1;
            %                 return; %You've found what you want, move on ...
        end;    %check if old confidence value in range for new confidence value
    end;    %new confidence levels
end;    %rows of data

%default (group_setting == 0)is to organise plots by expertise group, with first plot experts
%and second non-experts. But if we flip group_setting to 1, then plots orgamised by
%CFMT+ median split with 0 low CFMT+ and 1 high CFMT+.
%It achieves this by replacing the expertise column in big_data with the
%cfmt_median_split column instead so everything it did do with expertise
%group it now does with CFMT+ median split.
if group_setting == 0;
    num_groups = 2; %default e.g., for experts versus non
    figure_titles = {'experts' 'non-experts'};
    %     line_colour = lines(num_groups);
    temp = gray(num_groups+1);  %Can't use white, drop it off the end
    line_colour = temp(1:num_groups,:);
    legend_incr = -.075;
    legend_locs = [0.3:legend_incr:.3+legend_incr*num_groups];
elseif group_setting == 1;
    big_data(:,13) = cfmt_median_split;
    %     figure_titles = {'Low CFMT+' 'High CFMT+'};
    num_groups = numel(split_values)-1;
    for i=1:num_groups;
%         figure_titles{i} = sprintf('CFMT+ %d to %d',round(split_values(i)),round(split_values(i+1)));
    figure_titles = {'Developmental prosopagnosia' 'Typical recognisers' 'Super recognisers'};
        %         line_colour = jet(num_groups);
        %         line_colour = lines(num_groups);
        temp = gray(num_groups+2);  %Can't use white, drop it off the end
        line_colour = temp([1 3 4],:);
        legend_incr = -.075;
        legend_locs = [0.25:legend_incr:.3+legend_incr*num_groups];
    end;
end;

h5 = figure; set(gcf,'Color',[1 1 1]); %To hold plots of pre-registered analyses, d prime and AUC.
h6 = figure; set(gcf,'Color',[1 1 1]); %to hold ROC plots
h7 = figure; set(gcf,'Color',[1 1 1]); %to hold hits/FAs & C plots

h3 = figure; set(gcf,'Color',[1 1 1]); %To hold confidence / proportion correct plots
jitter = randn(1,num_groups)*.01;   %each group will have same displacement in every condition for every marker
% h4 = figure; set(gcf,'Color',[1 1 1]); %To hold quartile plots


out_data = [];  %anova (averaged over trial data)
out_data_mm = [];   %mixed model (with trial data)

%This is for accumulating the conf-acc slopes for each participant so stats
%can be applied to them later.
conf_accu_slope_data = [];
conf_accu_slope_data_it = 1;

for group = 1:num_groups;    %either expert groups or CFMT+ groups, depends on group_setting above
    
    clear data old_data *hits* *fas* this* *c_* *ss* *auc* *dp_*
    %
    %narrow down to just one group
    data = big_data(find(big_data(:,13) == group-1),:);
    
    num_subs = numel(unique(data(:,2)));
    disp(sprintf('There are %d subjects in group %d',num_subs,group));
    
    %make a new variable for hits
    %if old and sub's answer matches correct answer, assign a 1 to 15th col of data
    data(find(data(:,8) == 0 & data(:,4) == data(:,8)),15) = 1;
    
    %average hits over old stimuli for each participant
    old_data = data(find(data(:,8) == 0),:);
    %average study car levels (all four) then subjects
    % [hits_ss_s hits_grps_s ns_s] = grpstats( old_data(:,15), [ old_data(:,11) old_data(:,2)], {'mean' 'gname' 'numel'});
    [hits_ss_s hits_grps_s ns_s] = grpstats( old_data(:,15), [ old_data(:,10) old_data(:,2)], {'mean' 'gname' 'numel'});
    hits_car_grps_s = str2num(cell2mat(hits_grps_s(:,1)));
    hits_car_ss_grps_s = str2num(cell2mat(hits_grps_s(:,2)));
    % %average test car levels then subjects
    % [hits_ss_t hits_grps_t ns_t] = grpstats( old_data(:,15), [ old_data(:,12) old_data(:,2)], {'mean' 'gname' 'numel'});
    % hits_car_grps_t = str2num(cell2mat(hits_grps_t(:,1)));
    % hits_car_ss_grps_t = str2num(cell2mat(hits_grps_t(:,2)));
    
    %make a new variable for false alarms
    %if new and sub's answer doesn't match correct answer, assign a 1 to 16th col of data
    data(find(data(:,8) == 1 & data(:,4) ~= data(:,8)),16) = 1;
    
    out_data_mm = [out_data_mm; data];
    
    %average false alarms over new stimuli for each participant
    new_data = data(find(data(:,8) == 1),:);
    %average car levels then subjects (fas can only be affected by car at test)
    [fas_ss fas_grps] = grpstats( new_data(:,16), [ new_data(:,12) new_data(:,2)], {'mean' 'gname'});
    fas_car_grps = str2num(cell2mat(fas_grps(:,1)));
    fas_car_ss_grps = str2num(cell2mat(fas_grps(:,2)));
    
    %deal with extreme values for d prime and c (Macmillan & Kaplan, 1985)
    N = 24;     %number of old/new stimuli
    hits_adj_ss_s = hits_ss_s;
    hits_adj_ss_s( find(hits_ss_s(:,1) == 0), 1) = 1/(2*N);
    hits_adj_ss_s( find(hits_ss_s(:,1) == 1), 1) = 1 - 1/(2*N);
    % hits_adj_ss_t = hits_ss_t;
    % hits_adj_ss_t( find(hits_ss_t(:,1) == 0), 1) = 1/(2*N);
    % hits_adj_ss_t( find(hits_ss_t(:,1) == 1), 1) = 1 - 1/(2*N);
    fas_adj_ss = fas_ss;
    fas_adj_ss( find(fas_ss(:,1) == 0), 1) = 1/(2*N);
    fas_adj_ss( find(fas_ss(:,1) == 1), 1) = 1 - 1/(2*N);
    
    
    %d prime - each of the four conditions has an average d prime and it's
    %computed by applying the test condition appropriate false alarm rate to
    %every hit rate. Not easy to do as automatically as before so let's loop
    %through every hit rate, check what its appropriate FA rate should be and
    %apply it.
    for trial = 1:size(hits_adj_ss_s,1);
        
        clear this_trial_condition this_trial_condition_FA_code;
        
        %check condition of this person's hit rate. Remember (0=car-car, 1=car-anticar, 2=anticar,car, 3=anticar-anticar)
        this_trial_subid = hits_car_ss_grps_s(trial,1);
        this_trial_condition = hits_car_grps_s(trial,1);
        %There are two test car conditions for FAs which are coded 0 or 1 -
        %which is appropriate for this trial?
        if this_trial_condition == 0 | this_trial_condition == 2; %tested as caricature
            this_trial_condition_FA_code = 0;
        else
            this_trial_condition_FA_code = 1;
        end;
        %search 2 test car conditions * 400 participants = 800 array for
        %correct adjusted false alarm rate to apply on this trial
        this_trial_FA(trial,1) = ...
            fas_adj_ss( ...
            find(...
            fas_car_grps == this_trial_condition_FA_code ...
            & fas_car_ss_grps == this_trial_subid ...
            ) ... %close find
            );      %close fas_adj_ss
        
        %apply FA to d prime computation
        dprime_ss_s(trial,1) = norminv(hits_adj_ss_s(trial,1),0,1) - norminv( this_trial_FA(trial,1) ,0,1);
        %and criterion
        c_ss_s(trial, 1) = (norminv(hits_adj_ss_s(trial,1),0,1) + norminv( this_trial_FA(trial,1),0,1))/0.5;
        
    end;    %loop through trial = 1600 = 400 participants*four car conditions
    % dprime_ss_s = norminv(hits_adj_ss_s,0,1) - norminv(fas_adj_ss,0,1);
    % %d prime test
    % dprime_ss_t = norminv(hits_adj_ss_t,0,1) - norminv(fas_adj_ss,0,1);
    %
    % %criterion study
    % c_ss_s = (norminv(hits_adj_ss_s,0,1) + norminv(fas_adj_ss,0,1))/0.5;
    % %criterion test
    % c_ss_t = (norminv(hits_adj_ss_t,0,1) + norminv(fas_adj_ss,0,1))/0.5;
    %
    
    %plot hits and false alarms
    % plot_cmap = hsv( numel(unique(hits_car_grps_s)));  %scale number of colours to be number of conditions used
    %     plot_cmap = [ hsv(2); hsv(2)*.5];  %scale number of colours to be number of conditions used
%     plot_cmap = [ repmat([.3 .3 .3; .7 .7 .7],2,1)];  %scale number of colours to be number of conditions used
    plot_cmap = [ repmat([0 0 1; 1 .5 0],2,1)];  %scale number of colours to be number of conditions used
    dot_colours = [0 0 0; 0 0 0; 0 0 0; 0 0 0];
    sw = 0.5;  %point spread width
    f_a = 0; %face alpha
    font_size = 12;
    xpos = [1.15 1.85 3.15 3.85 5.15 5.85]; %The first 4 are 2 study cars * 2 test cars. The last two are false alarm bars (test cars) used for old responses / hits plots only
    bar_width = .6;
    %     h1 = figure; set(gcf,'Color',[1 1 1],'Name',figure_titles{group});
    
    %2024, 4th April: Below, the bar order becomes the list of four conditions in column 10
    %by default, which confused me and led me to mislabel the axes. I'd
    %put test caricature on x axis and dodge the study caricature bars and 
    %have the labels be alphabetically ordered, so A then C.
%     bar_order = [4 2 3 1];   %test caricature on x axis, numbers here are the new, desired x axis positions
    bar_order = [4 3 2 1];   %study caricature on x axis, numbers here are the new, desired x axis positions
    bar_order_fa = [2 1]; %false alarms only have test conditions and would appear as car (0) then anti car (1) according to column 12. I want alphabetic order same as hits so I will create new bar order.

    %plot study
    for i=1:4;  %four car conditions and bars

        this_it = bar_order(i);
        
        %hits sub datapoints
        figure(h7);
        
        if groups_in_cols == 1;
            subplot(2,num_groups,num_groups+group);
        else;
            subplot(num_groups,3,group*3-1);
        end;
        
        %accumulate data so we can do t-tests and draw on significant results later
        this_hits_data = hits_ss_s(find(hits_car_grps_s(:,1) == i-1),1);
        hits_for_ttests(:,i) = this_hits_data;
        
        %  %hits bars
        handles = plotSpread(this_hits_data, ...
            'xValues',xpos(this_it),'distributionColors',plot_cmap(this_it,:),'distributionMarkers','.', 'spreadWidth', sw);
        
        bar(xpos(this_it), mean(this_hits_data), ...
            'FaceColor',[1 1 1],'FaceAlpha',f_a,'EdgeColor',[0 0 0], 'BarWidth', bar_width );
        
%         set(gca...
%             ,'FontSize',12 ...
%             ,'FontName','Arial' ...
%             ,'XTick',[1.5 3.5] ...
%             ,'YTick',[0:.2:1] ...
%             , 'XTickLabel',{'Anticaricature' 'Caricature' } ...
%             );
%         xlim([.7 4.3]);
%         ylim([0 1]);
%         ylabel('"Old" responses');
%         xlabel('Study caricature');
%         box off;
        
%         figure(h7);
%         
%         if groups_in_cols == 1;
%             subplot(3,num_groups,2*num_groups+group);
%         else;
%             subplot(num_groups,3,group*3);
%         end;
        %false alarms have only two conditions
        if i == 1 | i == 2;
            %fas sub datapoints
            %             subplot(2,2,2);

            this_it_fa = bar_order_fa(i);
            
            %accumulate data so we can do t-tests and draw on significant results later
            this_fas_data = fas_ss(find(fas_car_grps(:,1) == i-1),1);
            fas_for_ttests(:,i) = this_fas_data;
            
            %  %fas bars
            handles = plotSpread(this_fas_data, ...
                'xValues',xpos(4+this_it_fa),'distributionColors',plot_cmap(this_it_fa,:),'distributionMarkers','.', 'spreadWidth', sw);

            bar(xpos(4+this_it_fa), mean(this_fas_data) , ...
                'FaceColor',[1 1 1],'FaceAlpha',f_a,'EdgeColor',[0 0 0], 'BarWidth', bar_width  );

        end;    %We only want to plot FAs if it's the first two i's.

        ylim([0 1]);
        %             xlim([.7 2.3]);
        set(gca...
            ,'FontSize',12 ...
            ,'FontName','Arial' ...
            ,'XTick',[1.5 3.5 5.5] ...
            ,'YTick',[0:.2:1] ...
            , 'XTickLabel',{'Anticaricature' 'Caricature' 'New'} ...
            );
        ylabel('"Old" responses');
        xlabel('Study Caricature');
        box off;

        %dprime datapoints

        %v6: you want column 2 and groupth row
        figure(h5);
        
        if groups_in_cols == 1;
            subplot(2,num_groups,num_groups+group);
        else
            subplot(num_groups,2,2*group);
        end;
        
        %accumulate data so we can do t-tests and draw on significant results later
        this_dp_data = dprime_ss_s(find(hits_car_grps_s(:,1) == i-1),1);
        dp_for_ttests(:,i) = this_dp_data;
        
        %  %dprime bars
        handles = plotSpread(this_dp_data , ...
            'xValues',xpos(this_it),'distributionColors',plot_cmap(this_it,:),'distributionMarkers','.', 'spreadWidth', sw);
        
        bar(xpos(this_it), mean(this_dp_data ) , ...
            'FaceColor',[1 1 1],'FaceAlpha',f_a,'EdgeColor',[0 0 0] , 'BarWidth', bar_width );
        set(gca...
            ,'FontSize',12 ...
            ,'FontName','Arial' ...
            ,'XTick',[1.5 3.5] ...
            ,'YTick',[-4:1:4] ...
            , 'XTickLabel',{'Anticaricature' 'Caricature' } ...
            );
        xlim([.65 4.35]);
        ylim([-2 4]);
        ylabel('d prime');
        xlabel('Study caricature');
        box off;
        
        if group == 1;
            text(1.5,3.9,'Test anticaricature','FontSize',12 ,'FontName','Arial','Color',plot_cmap(1,:));
            text(1.5,3.4,'Test caricature','FontSize',12 ,'FontName','Arial','Color',plot_cmap(2,:));
        end;
        
        
        %c datapoints
        figure(h7);
        
        if groups_in_cols == 1;
            subplot(2,num_groups,group);
        else
            subplot(num_groups,3,group*3-2);
        end;
        hold on;
        %         figure(h1);
        %         subplot(2,2,4);
        
        %  %c bars
        
        this_c_data = c_ss_s(find(hits_car_grps_s(:,1) == i-1),1);
        c_for_ttests(:,i) = this_c_data;
        
        handles = plotSpread(this_c_data, ...
            'xValues',xpos(this_it),'distributionColors',plot_cmap(this_it,:),'distributionMarkers','.', 'spreadWidth', sw);
        
        bar(xpos(this_it), mean(this_c_data) , ...
            'FaceColor',[1 1 1],'FaceAlpha',0,'EdgeColor',[0 0 0], 'BarWidth', bar_width );
        set(gca...
            ,'FontSize',12 ...
            ,'FontName','Arial' ...
            ,'XTick',[1.5 3.5] ...
            ,'YTick',[-10:5:10] ...
            , 'XTickLabel',{'Anticaricature' 'Caricature'} ...
            );    ylabel('Criterion');
        xlim([.7 4.3]);
        ylim([-10 10]);
        xlabel('Study caricature');
        box off;
        
        if group == 1;
            text(2,8.5,'Test anticaricature','FontSize',12 ,'FontName','Arial','Color',plot_cmap(1,:));
            text(2,6,'Test caricature','FontSize',12 ,'FontName','Arial','Color',plot_cmap(2,:));
        end;
        
    end;    %plot each bar/car level
    
    %Now that you've accumulated the data for the four conditions into neat
    %little matriuces, run t-tests on them
    pair_indices = nchoosek([1:4],2);
%     pair_indices = nchoosek(bar_order,2);
    h=nan(size( pair_indices,1),4);
    p=nan(size( pair_indices,1),4);
    %where to put lines, xpos already gives x positions
    ypos_hits = [1.2:-.033:1];  %can use for AUC too
    %     ypos_fas = [1.2:-.033:1];
    ypos_dp = [4.5:-.083:4];
    ypos_c = [12:-.5:9];
    for i=1:size( pair_indices,1);

        %c
        [h(i,4),p(i,4)] = ttest( diff( [c_for_ttests(:,pair_indices(i,1)) c_for_ttests(:,pair_indices(i,2))]')' );
        if p(i,4)*size( pair_indices,1) < .05;
            figure(h7);
            if groups_in_cols == 1;
                subplot(2,num_groups,group);
            else
                subplot(num_groups,3,group*3-2);
            end;
            line([xpos(bar_order(pair_indices(i,1))) xpos(bar_order(pair_indices(i,2)))],[ypos_c(i) ypos_c(i)],'color',[0 0 0],'lineWidth',1);
            ylim([-8.1 ypos_c(1)]);
        end;    %sign at Bonferroni rate?
        
        %hits
        [h(i,1) p(i,1) ] = ttest( diff( [hits_for_ttests(:,pair_indices(i,1)) hits_for_ttests(:,pair_indices(i,2))]' )' );  %compute test
        if p(i,1)*size( pair_indices,1) < .05;
            figure(h7);
            if groups_in_cols == 1;
                subplot(2,num_groups,num_groups+group);
            else;
                subplot(num_groups,3,group*3-1);
            end;
            line([xpos(bar_order(pair_indices(i,1))) xpos(bar_order(pair_indices(i,2)))],[ypos_hits(i) ypos_hits(i)],'color',[0 0 0],'lineWidth',1);
            ylim([0 ypos_hits(1)]);
        end;    %sign at Bonferroni rate?
   
        %dprime
        [h(i,3),p(i,3)] = ttest( diff( [dp_for_ttests(:,pair_indices(i,1)) dp_for_ttests(:,pair_indices(i,2))]')' );
        if p(i,3)*size( pair_indices,1) < .05;
            figure(h5);
            if groups_in_cols == 1;
                subplot(2,num_groups,num_groups+group);
            else
                subplot(num_groups,2,2*group);
            end;
            line([xpos(bar_order(pair_indices(i,1))) xpos(bar_order(pair_indices(i,2)))],[ypos_dp(i) ypos_dp(i)],'color',[0 0 0],'lineWidth',1);
             ylim([-2 ypos_dp(1)]);
        end;    %sign at Bonferroni rate?

    end;    %pairs (i)

    %only one test between the two test car conditions so no need for loop
    [h_fa,p_fa] = ttest( diff( [fas_for_ttests(:,1) fas_for_ttests(:,2)]')' );
    if p_fa < .05;
        figure(h7);
        if groups_in_cols == 1;
            subplot(2,num_groups,num_groups+group);
        else;
            subplot(num_groups,3,group*3-1);
        end;
        line([xpos(5) xpos(6)],[1.05 1.05],'color',[0 0 0],'lineWidth',1);
%         ylim([0 1.05]);
    end;    %sign at Bonferroni rate?

    %loop through tests and paint lines on plots



    
    
    
    
    %now working on the ROC
    
    %need to loop through confidence levels - there are 11 levels and I will need to collapse some for low N in individual subjects
    %11 is divisible by nothing
    collapse_levels = 1;     %make groups out of the 11 that have this many elements in them (e.g., every two)
    all_levels = [100:-10:0];
    
    %at study
    for i=1:ceil(numel(all_levels)/collapse_levels);  %the last confidence level will have fewer values
        
        %what confidence levels should I consider this time?
        these_levels = all_levels(1:collapse_levels*i-1);
        
        %hits
        
        %if old and sub's answer matches correct answer, and confidence in range, assign a 1
        hits_temp_s = zeros(size(data,1),1);
        hits_temp_s(find(data(:,8) == 0 & data(:,4) == data(:,8) & ismember( data(:,5), these_levels )),:) = 1;
        
        %this should give you an average of the old data only separately for car levels and subjects
        [hits_ss_temp_s(:,i) hits_grps_temp_s] = grpstats( hits_temp_s(find(data(:,8) == 0),:), ...
            [ data(find(data(:,8) == 0),10) data(find(data(:,8) == 0),2)], ...
            {'mean' 'gname'});
        hits_grps_temp_s = str2num(cell2mat(hits_grps_temp_s(:,1)));   %get subjects ids
        
        %now repeat average, but separately for only car levels this time
        %when you grab confidence interval, though, it is guaranteed to be over subjects, which you need
        [hits_car_s(:,i) hits_car_ci_temp_s] = grpstats( hits_ss_temp_s(:,i), ...
            [ hits_grps_temp_s ], ...
            {'mean' 'meanci'});
        %     hits_car_ci_s(:,i) = hits_car_s(:,i) - hits_car_ci_temp_s(:,1);
        
        %false alarms
        
        %if new and sub's answer doesn't match correct answer, and confidence in range, assign a 1
        fas_temp = zeros(size(data,1),1);
        fas_temp(find(data(:,8) == 1 & data(:,4) ~= data(:,8) & ismember( data(:,5), these_levels )),:) = 1;
        
        %this should give you an average of the new data only separately for car levels and subjects
        [fas_ss_temp(:,i) fas_grps_temp] = grpstats( fas_temp(find(data(:,8) == 1),:), ...
            [ data(find(data(:,8) == 1),12) data(find(data(:,8) == 1),2)], ...
            {'mean' 'gname'});
        fas_grps_temp = str2num(cell2mat(fas_grps(:,1)));   %get subjects ids
        
        %now repeat average, but separately for only car levels this time
        %when you grab confidence interval, though, it is guaranteed to be over subjects, which you need
        [fas_car(:,i) fas_car_ci_temp] = grpstats( fas_ss_temp(:,i), ...
            [ fas_grps_temp ], ...
            {'mean' 'meanci'});
        %     fas_car_ci(:,i) = fas_car(:,i) - fas_car_ci_temp(:,1);
        
    end;   %loop through confidence levels
    
    %         h2 = figure; set(gcf,'Color',[1 1 1],'Name',figure_titles{group});
    
    %now make ROC plot
    figure(h6);
    
    if groups_in_cols == 1;
        subplot(1,3,group);
    else
        subplot(3,1,group);
    end;
    
    plot([0 1],[0 1],'Color',[0 0 0],'LineWidth',2); hold on;
    %     xlim([0 max(max(fas_car))]); ylim([0 max(max(hits_car_s))]);
    xlim([0 .65]); ylim([0 .85]);
    set(gca,'FontSize',12,'FontName','Arial');
    ylabel('Cumulative hit rate');
    xlabel('Cumulative false alarm rate');
    box off;
    %(0=car-car, 1=car-anticar, 2=anticar,car, 3=anticar-anticar)
    plot(fas_car(2,:)', hits_car_s(4,:)','Marker','o','MarkerFaceColor',plot_cmap(1,:),'MarkerEdgeColor',plot_cmap(1,:),'Color',plot_cmap(1,:),'LineStyle',':','LineWidth',4,'MarkerSize',4);
    plot(fas_car(1,:)', hits_car_s(3,:)','Marker','o','MarkerFaceColor',plot_cmap(2,:),'MarkerEdgeColor',plot_cmap(2,:),'Color',plot_cmap(2,:),'LineStyle',':','LineWidth',4,'MarkerSize',4);
    plot(fas_car(2,:)', hits_car_s(2,:)','Marker','o','MarkerFaceColor',plot_cmap(1,:),'MarkerEdgeColor',plot_cmap(1,:),'Color',plot_cmap(1,:),'LineStyle','-','LineWidth',1,'MarkerSize',3);
    plot(fas_car(1,:)', hits_car_s(1,:)','Marker','o','MarkerFaceColor',plot_cmap(2,:),'MarkerEdgeColor',plot_cmap(2,:),'Color',plot_cmap(2,:),'LineStyle','-','LineWidth',1,'MarkerSize',3);

    % shadedErrorBar hits_car_s', fas_car')
    
    %now loop through subjects and compute AUC for each
    for sub=1:num_subs;
        
        %         %area under diagonal
        %         auc_diag_s = trapz([0 1],[0 1]);
        
        %get sub specific ROC
        this_hits_0_s = [hits_ss_temp_s(sub,:) 1];
        this_hits_1_s = [hits_ss_temp_s(sub+num_subs,:) 1];
        this_hits_2_s = [hits_ss_temp_s(sub+2*num_subs,:) 1];
        this_hits_3_s = [hits_ss_temp_s(sub+3*num_subs,:) 1];
        
        this_fas_0 = [fas_ss_temp(sub,:) 1];
        this_fas_1 = [fas_ss_temp(sub+num_subs,:) 1];
        
        %area between curve and x axis
        auc_s(sub,1) = trapz(this_fas_0,this_hits_0_s);
        auc_s(sub,2) = trapz(this_fas_1,this_hits_1_s);
        auc_s(sub,3) = trapz(this_fas_0,this_hits_2_s);
        auc_s(sub,4) = trapz(this_fas_1,this_hits_3_s);
        
        
        %         %You want AUC between curve and diagonal, not x axis, so rotate
        %         %points in curve -45 degrees so diagonal becomes new x axis
        %         theta = -pi/4;  %-45 degree angle
        %         R = [cos(theta) -sin(theta); sin(theta) cos(theta)];    %rotation matrix
        %
        %         %compute areas for rotated curves
        %         %(0=car-car, 1=car-anticar, 2=anticar,car, 3=anticar-anticar)
        %         %fas are x and hits are y
        %         rotated = R*[this_fas_0; this_hits_0_s]; %this needs to be x (fas) in first row and y (hits in second row), which will also be true of result
        %         auc_s(sub,1) = trapz( rotated(1,:), rotated(2,:) ); %trapz wants x then y, which means hits then fas, which means first row then recond
        %         rotated = R*[this_fas_1; this_hits_1_s];
        %         auc_s(sub,2) = trapz( rotated(1,:), rotated(2,:) );
        %         rotated = R*[this_fas_0; this_hits_2_s];
        %         auc_s(sub,3) = trapz( rotated(1,:), rotated(2,:) );
        %         rotated = R*[this_fas_1; this_hits_3_s];
        %         auc_s(sub,4) = trapz( rotated(1,:), rotated(2,:) );
        
        %wrong wrong wrong
        %         %(0=car-car, 1=car-anticar, 2=anticar,car, 3=anticar-anticar)
        %         auc_s(sub,1) = trapz(this_fas_0,this_hits_0_s) - auc_diag_s;
        %         auc_s(sub,2) = trapz(this_fas_1,this_hits_1_s) - auc_diag_s;
        %         auc_s(sub,3) = trapz(this_fas_0,this_hits_2_s) - auc_diag_s;
        %         auc_s(sub,4) = trapz(this_fas_1,this_hits_3_s) - auc_diag_s;
        
    end;    %loop through subs to compute AUC
    if group == num_groups;
%         legend chance C-C C-A A-C A-A; legend boxoff;
        legend chance A-A A-C C-A C-C; legend boxoff;

    end;
    
    
    %make auc plot
    figure(h5);
    if groups_in_cols == 1;
        subplot(2,num_groups,group);
    else;
        subplot(num_groups,2,2*group-1);
        %     subplot(2,3,group);
    end;
    
    for i=1:4;

        this_it = bar_order(i);
        
        handles = plotSpread(auc_s(:,i), ...
            'xValues',xpos(this_it),'distributionColors',plot_cmap(this_it,:),'distributionMarkers','.', 'spreadWidth', sw);
        
        bar(xpos(this_it), mean(auc_s(:,i)) , ...
            'FaceColor',[1 1 1],'FaceAlpha',0,'EdgeColor',[0 0 0], 'BarWidth', bar_width );

    end;    %car levels
    
    %run pairwise tests and plot significance lines
    for i=1:size( pair_indices,1);
        [h,p] = ttest( diff( [auc_s(:,pair_indices(i,1)) auc_s(:,pair_indices(i,2))]')' );
        if p*size( pair_indices,1) < .05;
            figure(h5);
            if groups_in_cols == 1;
                subplot(2,num_groups,group);
            else;
                subplot(num_groups,2,2*group-1);
                %     subplot(2,3,group);
            end;
            line([xpos(bar_order(pair_indices(i,1))) xpos(bar_order(pair_indices(i,2)))],[ypos_hits(i) ypos_hits(i)],'color',[0 0 0],'lineWidth',1);
%             ylim([0.2 ypos_hits(1)]);
            ylim([0 ypos_hits(1)]);

        end;    %sign at Bonferroni rate?
    end;    %pairs (i)
    
    
    xlim([.65 4.35])
    %     ylim([min(min(auc_s)) max(max(auc_s))]);
%     ylim([.2 1]);
    set(gca,'FontSize',12,'FontName','Arial','XTick',[1.5 3.5],'YTick',[0:.2:1], 'XTickLabel',{'Anticaricature' 'Caricature' });
    ylabel('Area under ROC (AUC)');
    xlabel('Study caricature');
    box off;
    
    %process anova output file
    
    %grab the cfmt+ scores for all participants in this group, sorted by
    %participant number
    cfmt = grpstats(data(:,14),data(:,2));
    
    out_temp = [ ...
        group*ones(num_subs,1) ...
        reshape(hits_ss_s,num_subs,4) ...
        reshape(fas_ss,num_subs,2) ...
        reshape(dprime_ss_s,num_subs,4) ...
        reshape(c_ss_s,num_subs,4) ...
        auc_s ...
        cfmt ...
        ];
    
    
    out_data = [out_data; out_temp];
    
    %so at this stage we have the matrix data for this group with 16
    %columns, with response in 4 (0 old 1 new) and correct response in 8 (0
    %old 1 new). Make an accuracy variable in 17 if they match
    data(find(data(:,4)==data(:,8)),17) = 1; %accuracy variable, 1 if an accurate response
    
    %Now you want a plot with prop correct responses per confidence
    %condition for each of the four caricature conditions. So loop through
    %caricature at study, then caricature at test, then confidence
    %I might have already done it above but am coming to code for first
    %time and easier to make new loop than engage with old code above
    
    %We want to collapse 11 confidence levels into (say) three, to view
    %slopes as function of accuracy more easily.
    num_confidence_levels_wanted = 3;   %settable by user
    %What are boundaries of new confidence levels?
    confidence_levels = unique(data(:,5));
    %     new_levels = ...
    %         min(confidence_levels): ...
    %         round(((max(confidence_levels)-min(confidence_levels))/num_confidence_levels_wanted)): ...
    %         max(confidence_levels)+1; %there should be num_confidence_levels_wanted + 1 elements, because all three new levels need both upper and lower boundaries
    
    new_levels = linspace(min(confidence_levels), max(confidence_levels)+1,4);
    
    %now make new variable in data (18th col) that encodes new c levels
    %from old ones using an expensive and not at all elegent method that
    %doesn't require any thinking
    for i=1:size(data,1);   %loop through each row of the data file
        %for each row, loop through the new confidence levels to check each
        for j=1:numel(new_levels) - 1;   %loop through lower bounds (don't need final upper bound
            if data(i,5) >= new_levels(j) & data(i,5) < new_levels(j+1);
                data(i,18) = j;
                %                 return; %You've found what you want, move on ...
            end;    %check if old confidence value in range for new confidence value
        end;    %new confidence levels
    end;    %rows of data
    
    %Now make a matrix that contains the codes you want to use for hits and
    %false alarms when computing accuracy in different conditions. Then I can use these for averaging with group stats.
    accur_conds = zeros(size(data,1),4);
    conf_and_acc = zeros(size(data,1),2);
    for trial=1:size(data,1);   %trials
        
        %col for CC
        if (data(trial,8) == 0 & data(trial,10) == 0) | .... %if old and condition 0 (CC) or ...
                (data(trial,8) == 1 & data(trial,12) == 0);  %new and test is C
            accur_conds(trial,1) = 1;
        end;
        %col for CA
        if (data(trial,8) == 0 & data(trial,10) == 1) | .... %if old and condition 1 (CA) or ...
                (data(trial,8) == 1 & data(trial,12) == 1);  %new and test is A
            accur_conds(trial,2) = 1;
        end;
        %col for AC
        if (data(trial,8) == 0 & data(trial,10) == 2) | .... %if old and condition 2 (AC) or ...
                (data(trial,8) == 1 & data(trial,12) == 0);  %new and test is C
            accur_conds(trial,3) = 1;
        end;
        if (data(trial,8) == 0 & data(trial,10) == 3) | .... %if old and condition 2 (AA) or ...
                (data(trial,8) == 1 & data(trial,12) == 1);  %new and test is A
            accur_conds(trial,4) = 1;
        end;    %if/then test for four conds
        
        %The above bit specifies the different conditions and the next bit
        %carves out proportion correct for three confidence levels, but
        %just get confidencve and accuracy into trial vectors for
        %regression later using this loop too
        conf_and_acc(trial,1) = data(trial,2);  %participant so can average 
        conf_and_acc(trial,2) = data(trial,5);  %this trial's confidence
        if data(trial,4) == data(trial,8);  %if participant (4th col) says the correct answer (8th col)
            conf_and_acc(trial,3) = 1; 
        end;
 
    end;    %trials
    
    
    %Get averages, CIs and N's for each of the four conditions in the new
    %vectors
    Parts = unique(data(:,2));
    for condition = 1:size(accur_conds,2);
        for conf = 1:num_confidence_levels_wanted;
            clear temp_Ps;
            for P = 1:numel(Parts);
                
                %get the accuracy data from the rows of data that match the codes
                %for the trialth condition
                relevant_indices = find( ...
                    data(:,2) ==Parts(P) & ...
                    accur_conds(:,condition)==1 & ...
                    data(:,18) == conf);
                if ~isempty(relevant_indices);  %If there are more than zero trials satisfying these conditions
                    temp_Ps(P,1) = nanmean(data(relevant_indices,17));
                else;
                    temp_Ps(P,1) = NaN;
                end;
            end;    %participants (P)
            
            %assign results collapsed over participants
            this_plot_data(condition,conf) = nanmean(temp_Ps);
            this_plot_n(condition,conf) = sum(~isnan(temp_Ps))/numel(Parts);
            this_plot_data_ci(condition,conf) = ...
                tinv(0.975,sum(~isnan(temp_Ps))-1)*(nanstd(temp_Ps)/sqrt(sum(~isnan(temp_Ps))));
            
        end;    %confidence levels (conf)
        
        %get confidence and accuracy data for each condition and run
        %regression on it
        fprintf(' ');
        clear this_cond_conf_and_acc;
        this_cond_conf_and_acc = conf_and_acc(find(accur_conds(:,condition)==1),:);
        this_cond_subs = unique(this_cond_conf_and_acc(:,1));
        for sub_i = 1:numel(this_cond_subs);  %loop through this group's subs
            clear this_cond_sub_conf_and_acc;
            this_cond_sub_conf_and_acc = this_cond_conf_and_acc(find(this_cond_conf_and_acc(:,1)==this_cond_subs(sub_i)),:);
            
            if sum(diff(this_cond_sub_conf_and_acc(:,2))) == 0; %if there's no varioation in scores, NaN it (this happens, I think, just once
                B = [NaN NaN];
            else;
                [B,dev,stats] = mnrfit(this_cond_sub_conf_and_acc(:,2),this_cond_sub_conf_and_acc(:,3)+1);
            slope_this_cond(sub_i,condition) = -B(2);   %Slopes seem to be upside down, I want them presented as predicting correct answers
            end;

            %accumulate subject data
            conf_accu_slope_data_temp(sub_i,1) = this_cond_subs(sub_i);
            conf_accu_slope_data_temp(sub_i,2) = group;
            conf_accu_slope_data_temp(sub_i,2+condition) =   -B(2);
            %             conf_accu_slope_data = [conf_accu_slope_data; [group this_cond_subs(sub_i) condition -B(2)]];  %accumulate results for further analysis
            
        end;    %loop through this group's subs
        
    end;    %loop through four conditions
    
    conf_accu_slope_data = [conf_accu_slope_data; conf_accu_slope_data_temp];

    figure(h3);
    %     subplot(1,2,group);
    hold on;%one plot per group
    
    for i=1:size(this_plot_data,1); %the four conditions
        
        clear these_markers*;
        
        %Base locations of markers and errors
        %jitter defined way at the top so it'll be outside the group loop
        %and apply to all groups
        incr = 0.8/size(this_plot_data,2);
        these_markers_xs = i:incr:i+incr*size(this_plot_data,2)-incr;
        these_markers_xs = these_markers_xs + jitter(group);
        
        %line
        plot( ...
            these_markers_xs, ...
            this_plot_data(i,:), ...
            'Marker', 'none', ...
            'Color',line_colour(group,:),...
            'LineWidth',2 ...
            );
        %                     i:incr:i+incr*size(this_plot_data,2)-incr, ...
        
        text(these_markers_xs(3)+.1,this_plot_data(i,3),sprintf('%2.2f',(nanmean(slope_this_cond(:,i)))),'Color',line_colour(group,:));        
        
        %error bars
        errorbar( ...
            these_markers_xs, ...  %x
            this_plot_data(i,:), ...                        %y
            this_plot_data_ci(i,:), ...                     %err (ci)
            'Color',line_colour(group,:),...
            'LineWidth',1 ...
            );
        %             i:incr:i+incr*size(this_plot_data,2)-incr, ...  %x
        
        %This code uses markers whose size is determined by the n in each
        %but in an absolute sense, that are not scaled by condition. So the
        %DP and super-recog end up invisibly tiny and the typical group is
        %really large. Comparisons between different confidence levels
        %within condition / group are challenging. Bubbleplot maybe better
        %for those.
        %         %add markers
        these_markers = this_plot_data(i,:);
        %         these_marker_sizes = this_plot_n(i,:);
        %         these_markers_xs = i:incr:i+incr*size(this_plot_data,2)-incr;
        %         reduce_marker_size = 10;    %factor by which to divide marker size
        %rescale to marker point range (assuming no proportions below .5)
        lower_bound = .5;   %what's the minimum value that should be the smallest marker size
        marker_max = 10;
        marker_min = 2;
        these_marker_sizes = ((this_plot_n(i,:) - lower_bound)/(1-lower_bound))*(marker_max-marker_min)+marker_min;
        for marker = 1:size(these_markers,2);
            
            %differently sized markers
            plot( ...
                these_markers_xs(marker), ...   %x
                these_markers(marker), ...      %y
                'Marker', 'o', ...
                'MarkerEdgeColor',[0 0 0], ...
                'MarkerFaceColor',line_colour(group,:), ...
                'MarkerSize',these_marker_sizes(marker), ...
                'Color',[0 0 0],...
                'LineWidth',1 ...
                );
            
        end;    %marker
        
    end;    %%the four conditions
    
    
    set(gca, ...
        'XTick',[1+incr:1:4+incr], ...
        'XTickLabel',{'C-C','C-A','A-C','A-A'}, ...
        'FontSize',12,  ...
        'FontName','Arial' ...
        );
    %     title(figure_titles{group}, ...
    %         'FontSize',12,  ...
    %         'FontName','Arial' ...
    %         );
    ylabel('proportion correct', ...
        'FontSize',12,  ...
        'FontName','Arial' ...
        );
    xlabel('Caricature conditions Study-Test', ...
        'FontSize',12,  ...
        'FontName','Arial' ...
        );
    ylim([0 1]);
    xlim([.5 5])
    
    text(2+incr,legend_locs(num_groups-group+1), figure_titles{group}, ...
        'Color',line_colour(group,:), ...
        'FontSize',12,  ...
        'FontName','Arial' ...
        );
    
end;    %Loop through groups

disp('audi5000');