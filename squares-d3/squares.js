function plot_it()  {
	predictions_data.forEach((d,i) => {
		d.index = i;
	});
	var shuffled_data = d3.shuffle(predictions_data);
	var num_predictions = 5000;
	predictions_data = [];
	for(var i = 0; i < num_predictions; i++)
		predictions_data.push(shuffled_data[i]);

	// width, height, number of squares per bin column, number of bins
	// var width = 2000, height = 1400;
	var width = 4000, height = 1400;
	d3.select('body').append('svg').attr('width', width).attr('height', height);
	// var pad = 40;
	var pad = 200;
	var actual_width = width-2*pad, actual_height = height-2*pad;
	// var squares_per_column = 14;
	var squares_per_column = 85;
	var n_bins = 10;

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
}
