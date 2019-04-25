function plot_it()  {


	////////////////////////////////////////////////////////////////////////////////////////////////////
	// squres 1
	////////////////////////////////////////////////////////////////////////////////////////////////////
	
	predictions_data.forEach((d,i) => {
		d.index = i;
	});
	var shuffled_data = d3.shuffle(predictions_data);
	var num_predictions = 5000;
	predictions_data = [];
	for(var i = 0; i < num_predictions; i++)
		predictions_data.push(shuffled_data[i]);

	// width, height, number of squares per bin column, number of bins
	var width = 1750, height = 600;	// 2000, 1400
	var chart2 = d3.select("#area1").append('svg').attr('width', width).attr('height', height);
	var pad = 20;
	var actual_width = width-2*pad, actual_height = height-2*pad;
	var squares_per_column = 5;	// 14
	var n_bins = 15;	// 15

	var cats = d3.set(predictions_data.map(d => d.gt)).values()
	cats.sort()
	var cat_to_ind = {};
	cats.forEach((d,i) => {
		cat_to_ind[d] = i;
	});

	// First step: we need to derive any necessary data for our vis. What we need here are two things:
	// 1) the bin number, given the score
	// 2) sorting the squares, based on false positives and true positives
	predictions_data.forEach(datum =>  {
		datum.bin = Math.floor(n_bins*(+(datum.score)));
		if(datum.bin == n_bins)
			datum.bin = n_bins-1;
	});

	// We NEST: group data by: prediction category, bin
	var nested_data = d3.nest()
		.key(d => d.pred)
		.key(d => d.bin)
		.sortValues((a,b) =>  {
			var a_gt_ind = a.gt==a.pred ? cats.length+1 : cat_to_ind[a.gt];
			var b_gt_ind = b.gt==b.pred ? cats.length+1 : cat_to_ind[b.gt];
			return a_gt_ind - b_gt_ind;
		})
		.entries(predictions_data)

	// Second step: we need to determine our data ranges -> unique items for categories, min/max for quantitative data
	var bins = [];
	for(var i = 0; i < n_bins; i++)
		bins.push(i);

	var bin_column_offsets = [];
	for(var i = 0; i < squares_per_column; i++)
		bin_column_offsets.push(i);

	// Third step: we setup our scales -> here, we have made decisions on our visual encodings, and we use scales to map from data to visual channel

	// first scale: map from predicted category to column (band scale)
	var cat_scale_x = d3.scaleBand().domain(cats).range([pad,actual_width]).padding(0.3);

	// second scale: map from bin id to row (band scale)
	var bin_scale_y = d3.scaleBand().domain(bins).range([actual_height,pad]).paddingOuter(0.02).paddingInner(0.07);
	var score_scale_y = d3.scaleLinear().domain([0,1]).range([actual_height,pad]);

	// third scale: map from bin column id to y-offset for a square (we'll use this to position for x-offsets too, but we could use a scale for this as well)
	var square_scale_y = d3.scaleBand().domain(bin_column_offsets).range([0,bin_scale_y.bandwidth()]).paddingInner(0.2);

	// fourth scale: map from predicted category to color (color scale)
	var cat_scale_color = d3.scaleOrdinal(d3.schemePaired);

	cats.forEach((d,i) => {
		var consistency_color = cat_scale_color(cat_to_ind[d]);
	});

	// Fourth step: data joins

	// first: make an empty selection, to setup the parent SVG element
	var empty_rects = d3.select('svg').selectAll('g')

	// first data join: on predictions
	var pred_data_join = empty_rects.data(nested_data, d => d.key);
	var pred_data_groups = pred_data_join.enter().append('g')
		.attr('transform', d => 'translate('+cat_scale_x(d.key)+','+0+')')

	// add in a line per-column
	pred_data_groups.append('line')
		.attr('x1',0).attr('x2',0)
		.attr('y1',pad+bin_scale_y.paddingOuter()*bin_scale_y.step()).attr('y2',actual_height-bin_scale_y.paddingOuter()*bin_scale_y.step())
		.attr('stroke', d => cat_scale_color(cat_to_ind[d.key]))
		.attr('stroke-width', 3)

	// and, show category labels
	pred_data_groups.append('text')
		.text(d => d.key)
		.attr('class', 'thecat')
		.attr('x', 0).attr('y', pad-3)
		.attr('text-anchor', 'middle').attr('font-size', 16)
		.attr('fill', d => cat_scale_color(cat_to_ind[d.key]))

	// second data join: on bins
	var bin_data_join = pred_data_groups.selectAll('g').data(d => d.values, d => d.key);
	var bin_data_groups = bin_data_join.enter().append('g').attr('transform', d => 'translate('+0+','+bin_scale_y(d.key)+')')

	// third data join: individual squares
	var rect_data_join = bin_data_groups.selectAll('g').data(d => d.values);
	var all_square_groups = rect_data_join.enter().append('g')

	// first: create transformation to position each square
	all_square_groups.attr('transform', (d,i) => {
		var x_off = Math.floor(i/squares_per_column);
		var square_offset_x = 3+x_off*square_scale_y.step();
		var y_off = i % squares_per_column;
		var square_offset_y = square_scale_y(y_off);
		return 'translate('+square_offset_x+','+square_offset_y+')';
	});

	// also - we want to distinguish false positives from true positives -> class it accordingly
	all_square_groups.attr('class', d => { return d.pred==d.gt ? 'tp' : 'fp' });

	// next: create rectangles, style them
	all_square_groups.append('rect')
		.attr('width', square_scale_y.bandwidth())
		.attr('height', square_scale_y.bandwidth())
		.attr('fill', (d,i) => cat_scale_color(cat_to_ind[d.gt]))

	var rel_stripe_position_1 = 7*square_scale_y.bandwidth()/10;
	var rel_stripe_position_2 = 3*square_scale_y.bandwidth()/10;
	// last: style false positives with striped lines
	d3.selectAll('.fp').append('line')
		.attr('stroke', 'white').attr('stroke-width', 1.2)
		.attr('x1', 0).attr('x2', rel_stripe_position_1)
		.attr('y1', rel_stripe_position_1).attr('y2', 0)
	d3.selectAll('.fp').append('line')
		.attr('stroke', 'white').attr('stroke-width', 1.2)
		.attr('x1', rel_stripe_position_2).attr('x2', square_scale_y.bandwidth())
		.attr('y1', square_scale_y.bandwidth()).attr('y2', rel_stripe_position_2)

	// left axis for predictions scale
	d3.select('svg').append('g').attr('transform', 'translate('+(pad)+','+(0)+')').call(d3.axisLeft(score_scale_y));

	// mouse over, mouse out events for plotting full scores
	d3.selectAll('.thecat').on('mouseover', function(cat_d,i)  {
		// go through each bin, and then grab all of the indexes from individual predictions
		var selected_scores = [];
		for(var i = 0; i < cat_d.values.length; i++)  {
			var bin_values = cat_d.values[i];
			for(var j = 0; j < bin_values.values.length; j++)
				selected_scores.push(scores_data[bin_values.values[j].index]);
		}

		// our simple line shape
		line_shape = d3.line()
			.x(d => cat_scale_x(d[0]))
			.y(d => score_scale_y(d[1]))

		// a data join on our selected scores...
		d3.select('svg').selectAll('lines').data(selected_scores).enter().append('path')
			.attr('class', 'scoreline')
			.attr('d', d => {
				// line expects an array, so we format our scores accordingly
				var score_line = [];
				cats.forEach((cat,i) => {
					score_line.push([cat,d[cat]]);
				});
				return line_shape(score_line);
			})
			.attr('fill', 'none')
			.attr('stroke', '#888888')
			.attr('stroke-opacity', '0.05')
			.attr('stroke-width', '2')
	});
	d3.selectAll('.thecat').on('mouseout', function(d,i)  {
		// in the above, we denote each path element with class scoreline, so it is trivial to remove them
		d3.selectAll('.scoreline').remove()
	});

	// mouse over, mouse out events for individual score plots
	d3.selectAll('.tp,.fp').on('mouseover', function(score,i)  {
		// grab prediction scores, format for line, similar to above...
		scores = scores_data[score.index];
		var line_datum = [];
		cats.forEach((cat,i) => {
			line_datum.push([cat,scores[cat]]);
		});

		line_shape = d3.line()
			.x(d => cat_scale_x(d[0]))
			.y(d => score_scale_y(d[1]))

		d3.select('svg').append('path').datum(line_datum)
			.attr('class', 'scoreline')
			.attr('d', d => line_shape(d))
			.attr('fill', 'none')
			.attr('stroke', '#888888')
			.attr('stroke-opacity', '0.6')
			.attr('stroke-width', '3')
	});
	d3.selectAll('.tp,.fp').on('mouseout', function(d,i)  {
		d3.selectAll('.scoreline').remove()
	});


	////////////////////////////////////////////////////////////////////////////////////////////////////
	// dash 2
	////////////////////////////////////////////////////////////////////////////////////////////////////

	var fData=[
		{State:'AL',freq:{low:4786, mid:1319, high:249}}
		,{State:'AZ',freq:{low:1101, mid:412, high:674}}
		,{State:'CT',freq:{low:932, mid:2149, high:418}}
		,{State:'DE',freq:{low:832, mid:1152, high:1862}}
		,{State:'FL',freq:{low:4481, mid:3304, high:948}}
		,{State:'GA',freq:{low:1619, mid:167, high:1063}}
		,{State:'IA',freq:{low:1819, mid:247, high:1203}}
		,{State:'IL',freq:{low:4498, mid:3852, high:942}}
		,{State:'IN',freq:{low:797, mid:1849, high:1534}}
		,{State:'KS',freq:{low:162, mid:379, high:471}}
		];

	var barColor = 'steelblue';
    function segColor(c){ return {low:"#807dba", mid:"#e08214",high:"#41ab5d"}[c]; }
	    
	    // compute total for each state.
	    fData.forEach(function(d){d.total=d.freq.low+d.freq.mid+d.freq.high;});
	    
	    // function to handle histogram.
	    function histoGram(fD){
	        var hG={},    hGDim = {t: 60, r: 0, b: 30, l: 0};
	        hGDim.w = 500 - hGDim.l - hGDim.r, 
	        hGDim.h = 300 - hGDim.t - hGDim.b;
	            
	        //create svg for histogram.
	        var hGsvg = d3.select("#area2").append("svg")
	            .attr("width", hGDim.w + hGDim.l + hGDim.r)
	            .attr("height", hGDim.h + hGDim.t + hGDim.b).append("g")
	            .attr("transform", "translate(" + hGDim.l + "," + hGDim.t + ")");

	        // create function for x-axis mapping.
	        var x = d3.scaleBand().rangeRound([0, hGDim.w], 0.1)
	                .domain(fD.map(function(d) { return d[0]; }));

	        // Add x-axis to the histogram svg.
	        hGsvg.append("g").attr("class", "x axis")
	            .attr("transform", "translate(0," + hGDim.h + ")")
	            .call(d3.axisBottom().scale(x));

	        // Create function for y-axis map.
	        var y = d3.scaleLinear().range([hGDim.h, 0])
	                .domain([0, d3.max(fD, function(d) { return d[1]; })]);

	        // Create bars for histogram to contain rectangles and freq labels.
	        var bars = hGsvg.selectAll(".bar").data(fD).enter()
	                .append("g").attr("class", "bar");
	        
	        //create the rectangles.
	        bars.append("rect")
	            .attr("x", function(d) { return x(d[0]); })
	            .attr("y", function(d) { return y(d[1]); })
	            .attr("width", x.bandwidth())
	            .attr("height", function(d) { return hGDim.h - y(d[1]); })
	            .attr('fill',barColor)
	            .on("mouseover",mouseover)// mouseover is defined below.
	            .on("mouseout",mouseout);// mouseout is defined below.
	            
	        //Create the frequency labels above the rectangles.
	        bars.append("text").text(function(d){ return d3.format(",")(d[1])})
	            .attr("x", function(d) { return x(d[0])+x.bandwidth()/2; })
	            .attr("y", function(d) { return y(d[1])-5; })
	            .attr("text-anchor", "middle");
	        
	        function mouseover(d){  // utility function to be called on mouseover.
	            // filter for selected state.
	            var st = fData.filter(function(s){ return s.State == d[0];})[0],
	                nD = d3.keys(st.freq).map(function(s){ return {type:s, freq:st.freq[s]};});
	               
	            // call update functions of pie-chart and legend.    
	            pC.update(nD);
	            leg.update(nD);
	        }
	        
	        function mouseout(d){    // utility function to be called on mouseout.
	            // reset the pie-chart and legend.    
	            pC.update(tF);
	            leg.update(tF);
	        }
	        
	        // create function to update the bars. This will be used by pie-chart.
	        hG.update = function(nD, color){
	            // update the domain of the y-axis map to reflect change in frequencies.
	            y.domain([0, d3.max(nD, function(d) { return d[1]; })]);
	            
	            // Attach the new data to the bars.
	            var bars = hGsvg.selectAll(".bar").data(nD);
	            
	            // transition the height and color of rectangles.
	            bars.select("rect").transition().duration(500)
	                .attr("y", function(d) {return y(d[1]); })
	                .attr("height", function(d) { return hGDim.h - y(d[1]); })
	                .attr("fill", color);

	            // transition the frequency labels location and change value.
	            bars.select("text").transition().duration(500)
	                .text(function(d){ return d3.format(",")(d[1])})
	                .attr("y", function(d) {return y(d[1])-5; });            
	        }        
	        return hG;
	    }
	    
	    // function to handle pieChart.
	    function pieChart(pD){
	        var pC ={},    pieDim ={w:250, h: 250};
	        pieDim.r = Math.min(pieDim.w, pieDim.h) / 2;
	                
	        // create svg for pie chart.
	        var piesvg = d3.select("#area2").append("svg")
	            .attr("width", pieDim.w).attr("height", pieDim.h).append("g")
	            .attr("transform", "translate("+pieDim.w/2+","+pieDim.h/2+")");
	        
	        // create function to draw the arcs of the pie slices.
	        var arc = d3.arc().outerRadius(pieDim.r - 10).innerRadius(0);

	        // create a function to compute the pie slice angles.
	        var pie = d3.pie().sort(null).value(function(d) { return d.freq; });

	        // Draw the pie slices.
	        piesvg.selectAll("path").data(pie(pD)).enter().append("path").attr("d", arc)
	            .each(function(d) { this._current = d; })
	            .style("fill", function(d) { return segColor(d.data.type); })
	            .on("mouseover",mouseover).on("mouseout",mouseout);

	        // create function to update pie-chart. This will be used by histogram.
	        pC.update = function(nD){
	            piesvg.selectAll("path").data(pie(nD)).transition().duration(500)
	                .attrTween("d", arcTween);
	        }        
	        // Utility function to be called on mouseover a pie slice.
	        function mouseover(d){
	            // call the update function of histogram with new data.
	            hG.update(fData.map(function(v){ 
	                return [v.State,v.freq[d.data.type]];}),segColor(d.data.type));
	        }
	        //Utility function to be called on mouseout a pie slice.
	        function mouseout(d){
	            // call the update function of histogram with all data.
	            hG.update(fData.map(function(v){
	                return [v.State,v.total];}), barColor);
	        }
	        // Animating the pie-slice requiring a custom function which specifies
	        // how the intermediate paths should be drawn.
	        function arcTween(a) {
	            var i = d3.interpolate(this._current, a);
	            this._current = i(0);
	            return function(t) { return arc(i(t));    };
	        }    
	        return pC;
	    }
	    
	    // function to handle legend.
	    function legend(lD){
	        var leg = {};
	            
	        // create table for legend.
	        var legend = d3.select("#area2").append("table").attr('class','legend');
	        
	        // create one row per segment.
	        var tr = legend.append("tbody").selectAll("tr").data(lD).enter().append("tr");
	            
	        // create the first column for each segment.
	        tr.append("td").append("svg").attr("width", '16').attr("height", '16').append("rect")
	            .attr("width", '16').attr("height", '16')
				.attr("fill",function(d){ return segColor(d.type); });
	            
	        // create the second column for each segment.
	        tr.append("td").text(function(d){ return d.type;});

	        // create the third column for each segment.
	        tr.append("td").attr("class",'legendFreq')
	            .text(function(d){ return d3.format(",")(d.freq);});

	        // create the fourth column for each segment.
	        tr.append("td").attr("class",'legendPerc')
	            .text(function(d){ return getLegend(d,lD);});

	        // Utility function to be used to update the legend.
	        leg.update = function(nD){
	            // update the data attached to the row elements.
	            var l = legend.select("tbody").selectAll("tr").data(nD);

	            // update the frequencies.
	            l.select(".legendFreq").text(function(d){ return d3.format(",")(d.freq);});

	            // update the percentage column.
	            l.select(".legendPerc").text(function(d){ return getLegend(d,nD);});        
	        }
	        
	        function getLegend(d,aD){ // Utility function to compute percentage.
	            return d3.format("%")(d.freq/d3.sum(aD.map(function(v){ return v.freq; })));
	        }

	        return leg;
	    }
	    
	    // calculate total frequency by segment for all state.
	    var tF = ['low','mid','high'].map(function(d){ 
	        return {type:d, freq: d3.sum(fData.map(function(t){ return t.freq[d];}))}; 
	    });    
	    
	    // calculate total frequency by state for all segment.
	    var sF = fData.map(function(d){return [d.State,d.total];});

	    var hG = histoGram(sF), // create the histogram.
	        pC = pieChart(tF), // create the pie-chart.
	        leg= legend(tF);  // create the legend.


}
