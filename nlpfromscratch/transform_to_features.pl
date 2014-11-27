#!/usr/local/bin/perl

use strict;

my %tag_map = ();

my @tags = `cat /data2/users/rahuljha/projects/nlp_from_scratch/code/senna/hash/pos.lst`;
map(chomp($_), @tags);

for(my $i=0;$i<=$#tags; $i++) {
    $tag_map{$tags[$i]} = $i+1;
}

my $log_file = "transform.log";
open LOG, ">$log_file" or die $!;

my @sets = 0..24;
@sets = map { $_ < 10 ? "0$_" : "$_"} @sets;

foreach my $set (@sets) {

    my $tagged_dir_name = "pos/tagged/$set";
    my $raw_dir_name = "pos/raw/wsj/$set";
    my $output_file_name = "pos/features/$set.txt";

    my @tagged_text = `cat $tagged_dir_name/*`;
    map(chomp($_), @tagged_text);

    my @labels = ();
    my @word_features = ();
    my @caps_features = ();

    my $senna_in = "/data2/users/rahuljha/projects/nlp_from_scratch/data/senna_in.txt";
    my $senna_out = "/data2/users/rahuljha/projects/nlp_from_scratch/data/senna_out.txt";

    open TEMP, ">./temp.txt" or die $!;
    open SENNA_IN, ">".$senna_in or die $!;

    foreach my $tagged_sent (@tagged_text) {
	my @words = split(/\s/, $tagged_sent);
	my @curr_labels = ();
	my $raw_sent = "";
	foreach my $word (@words) {
	    my ($w, $l) = split(/\//, $word);
	    $raw_sent .= "$w ";
	    my @entry = ($w, $tag_map{$l});
	    push(@curr_labels, \@entry);
	}

	print TEMP $tagged_sent."\n";
	print SENNA_IN $raw_sent."\n";
	push(@labels, \@curr_labels);
    }

    close TEMP;
    close SENNA_IN;

    `cd /data2/users/rahuljha/projects/nlp_from_scratch/code/senna/; ./senna < $senna_in > $senna_out`;

    open SENNA_OUT, $senna_out or die $!;

    while(<SENNA_OUT>) {
	chomp($_);
	next if $_ =~ m/^\s*$/;
	my @curr_word_feats = ();
	my @curr_caps_feats = ();
	my @feats = split(/\s/, $_);
	foreach my $f (@feats) {
	    my ($word, $wid, $cid) = split(/:/, $f);
	    my @w_entry = ($word, $wid);
	    my @c_entry = ($word, $cid);
	    push(@curr_word_feats, \@w_entry);
	    push(@curr_caps_feats, \@c_entry);
	}
	push(@word_features, \@curr_word_feats);
	push(@caps_features, \@curr_caps_feats);
    }

    close SENNA_OUT;


    my @features_with_labels = ();

    for(my $i=0; $i<=$#labels;$i++) {
	my @curr_feats_with_labels = ();

	my $sent_labels = $labels[$i];
	my $sent_word_feats = $word_features[$i];
	my $sent_caps_feats = $caps_features[$i];

	my $sent_labels_len = @$sent_labels;
	my $offset = 0;
	for(my $j=0; $j<$sent_labels_len; $j++) {
	    my $curr_label = $sent_labels->[$j];
	    my $curr_word_feat = $sent_word_feats->[$j+$offset];
	    my $curr_caps_feat = $sent_caps_feats->[$j+$offset];

	    if(($curr_label->[0] eq $curr_word_feat->[0]) && ($curr_label->[0] eq $curr_caps_feat->[0])) {
		my @vals = ($curr_word_feat->[1], $curr_caps_feat->[1], $curr_label->[1]);
		push(@curr_feats_with_labels, \@vals);
	    } else {
		print LOG "Mismatch in tokenization: orig -> ".$curr_label->[0]." senna ->".$curr_word_feat->[0]."\n";
		if($curr_word_feat->[0] eq "-") {
		    $offset = 0;
		    while((($j+$offset+1) < $sent_labels_len) && ($sent_word_feats->[$j+1+$offset]->[0] ne $sent_labels->[$j+1]->[0])) {
			$offset++;
		    }
		}
	    }
	}
	push(@features_with_labels, \@curr_feats_with_labels);
	
    }

    open OUTPUT, ">".$output_file_name or die $!;

    my $window_size = 2;
    my $stride = 2 * $window_size;
    for(my $i=0;$i<= $#features_with_labels;$i++) {
	my $curr_feats_with_labels = $features_with_labels[$i];

	push(@$curr_feats_with_labels, [1738,1,1]);
	push(@$curr_feats_with_labels, [1738,1,1]);
	unshift(@$curr_feats_with_labels, [1738,1,1]);
	unshift(@$curr_feats_with_labels, [1738,1,1]);
	my $len = @$curr_feats_with_labels;
	for(my $j=0; $j <= $len-1-$stride; $j++) {
	    my $label = $curr_feats_with_labels->[$j+$window_size]->[2];
	    for(my $k=0; $k<($stride+1); $k++) {
		if($curr_feats_with_labels->[$j+$k]->[0] == 0) {
		    $curr_feats_with_labels->[$j+$k]->[0] = 1739;
		}
		print OUTPUT $curr_feats_with_labels->[$j+$k]->[0]." ";
	    }
	    for(my $k=0; $k<($stride+1); $k++) {
		print OUTPUT $curr_feats_with_labels->[$j+$k]->[1]." ";
	    }
	    print OUTPUT $label."\n";
	}
    }

    close OUTPUT;
    print "done for $set\n";
}

close LOG;
