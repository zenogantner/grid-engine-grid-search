#!/usr/bin/perl

# (c) 2008, 2009, 2010 by Zeno Gantner <zeno.gantner@gmail.com>

# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MyMediaLite.  If not, see <http://www.gnu.org/licenses/>.

use strict;
use warnings;
binmode STDOUT, ':utf8';
binmode STDERR, ':utf8';

use Carp;
use English qw( -no_match_vars );
use File::Slurp;
use Getopt::Long;
use Regexp::Common;

my $DEFAULT_N_OF_FOLDS = 10;
my $DEFAULT_TOP_N      = 5;
my $verbose            = 0;

# TODO: eps, shrinking; more kernels and svm types

GetOptions(
	   'verbose+'           => \$verbose,
	   'help'               => \(my $help               = 0),

	   'result-directory=s' => \(my $result_directory   = '.'),

	   'job-name=s'         => \(my $job_name           = 'LIBSVM hyperparameter search'),
	   'email=s'            => \(my $email              = ''),
	   'priority=i'         => \(my $priority           = -1),

	   'log-c-min=i'        => \(my $log_c_min          = -5),
	   'log-c-max=i'        => \(my $log_c_max          = 15),
	   'log-g-min=i'        => \(my $log_g_min          = -15),
	   'log-g-max=i'        => \(my $log_g_max          = 3),
	   'log-w-min=i'        => \(my $log_w_min          = -5),
	   'log-w-max=i'        => \(my $log_w_max          = 15),
	   'stepsize=f'         => \(my $stepsize           = 2),
	   'log-c-values=s'     => \(my $log_c_values       = ''),
	   'log-g-values=s'     => \(my $log_g_values       = ''),
	   'log-w-values=s'     => \(my $log_w_values       = ''),

	   'folds=i'            => \(my $n_of_folds         = $DEFAULT_N_OF_FOLDS),

	   'check-results'      => \(my $check_results      = 0),
	   'measure=s'          => \(my $measure            = 'Cross Validation Accuracy'),
	   'top-n=i'            => \(my $top_n              = $DEFAULT_TOP_N),
	   'show-progress!'     => \(my $show_progress      = 1),
	   'suggest-interval'   => \(my $suggest_interval   = 0),

	   'executable=s'       => \(my $executable         = "$ENV{HOME}/src/libsvm/svm-train"),
	   'immediate-start'    => \(my $immediate_start    = 0),
	   'suppress-warnings'  => \(my $suppress_warnings  = 0),

	  ) or usage(-1);

if ($help) {
    usage(0);
}

if ($check_results) {
    check_results($result_directory);
    exit 0;
}

my $training_file = '';
if (scalar @ARGV > 0) {
    $training_file = $ARGV[0];
    if (! -e $training_file) {
	croak "Training file '$training_file' does not exist";
    }
}
else {
    print "Please give a training file\n\n";
    usage(-1);
}

if (! -e $executable) {
    croak "Exectuable '$executable' does not exist";
}

# TODO: generalize
my @log_c_values = ();
if ($log_c_values) {
    @log_c_values = split ' ', $log_c_values;
}
else {
    for (my $log_c = $log_c_min; $log_c <= $log_c_max; $log_c += $stepsize) {
	push @log_c_values, $log_c;
    }
}
my @log_w_values = ();
if ($log_w_values) {
    @log_w_values = split ' ', $log_w_values;
}
else {
    for (my $log_w = $log_w_min; $log_w <= $log_w_max; $log_w += $stepsize) {
	push @log_w_values, $log_w;
    }
}
my @log_g_values = ();
if ($log_g_values) {
    @log_g_values = split ' ', $log_g_values;
}
else {
    for (my $log_g = $log_g_min; $log_g <= $log_g_max; $log_g += $stepsize) {
	push @log_g_values, $log_g;
    }
}

my $qsub_command = "qsub -cwd -j n -b y -p $priority";
if ($email) {
     $qsub_command .= " -M $email -m beas";
}

print "Training file: $training_file\n";
print 'log C:         ' . (join ', ', @log_c_values) . "\n";
print 'log gamma:     ' . (join ', ', @log_g_values) . "\n";
print 'log w:         ' . (join ', ', @log_w_values) . "\n";
print "$n_of_folds-fold CV on all parameter combinations\n";
if (! $immediate_start) {
    print "Hit ENTER to proceed, Ctrl+C to abort\n";
    <STDIN>;
}

foreach my $log_c (@log_c_values) {
    foreach my $log_g (@log_g_values) {
	foreach my $log_w (@log_w_values) {
	    my $c = 2 ** $log_c;
	    my $g = 2 ** $log_g;
	    my $w = 2 ** $log_w;

	    my $job_arguments ="-s 0 -t 2 -v $n_of_folds -c $c -g $g -w1 $w -w-1 1 $training_file";
	    print "$executable $job_arguments\n";
	    system qq(echo "$job_arguments" > $result_directory/result-${log_c}-${log_g}-${log_w});
	    system qq($qsub_command -o $result_directory/result-${log_c}-${log_g}-${log_w} -N "$job_name ${log_c}-${log_g}-${log_w}" $executable $job_arguments);
	}
    }
}


# TODO: generalize
sub check_results {
    my ($dir) = @_;

    my @files = glob "$dir/result-*";

    my %measure = ();
    my %params  = ();

    my $finished_folds = 0;

  FILE:
    foreach my $file (@files) {
        print "Checking $file ...\n" if $verbose;
        my @lines = read_file($file);

	next FILE if scalar @lines == 1;

        # get parameters
        my $first_line = shift @lines; chomp $first_line;
        if ($first_line =~ /-v (\d+) -c ($RE{num}{real}) -g ($RE{num}{real}) -w1 ($RE{num}{real})/) {
            $n_of_folds = $1;
            my $c       = $2;
	    my $g       = $3;
	    my $w       = $4;

	    if ($file =~ /result-($RE{num}{real})-($RE{num}{real})-($RE{num}{real})$/) {
		my $log_c = $1;
		my $log_g = $2;
		my $log_w = $3;
		$params{$file} = {
				  c    => $c,
				  g    => $g,
				  w    => $w,
				  logC => $log_c,
				  logG => $log_g,
				  logW => $log_w,
				 };
	    }
	    else {
		croak "Could not parse result file name: '$file'";
	    }
        }
        else {
            carp "Could not parse first line of $file: '$first_line'";
	    next FILE;
        }

      LINE:
        foreach my $line (@lines) {
	    next LINE if ! $line =~ m/\n$/;
            chomp $line;
	    next LINE if   $line =~ m/^\s*$/;

            if ($line =~ /^$measure = ($RE{num}{real})/) {
                $measure{$file} = $1;
            }
	    elsif ($line =~ /^\.+/) {         # SVM training output
	    }
	    elsif ($line =~ /^\*+/) {         # SVM training output
	    }
            elsif ($line =~ /^nSV =/) { # SVM training output
	    }
            elsif ($line =~ /^Total nSV =/) { # SVM training output
	    }
            elsif ($line eq 'Warning: reaching max number of iterations') { # SVM training output
	    }
            elsif ($line =~ /^optimization finished/) {                     # SVM training output
                $finished_folds++;
	    }
            elsif ($line =~ /^obj =/) {
	    }
            elsif ($line =~ /^Cross Validation Accuracy/) {
	    }
            elsif ($line =~ /^(?=real|user|sys)/) {
            }
	    elsif ($suppress_warnings && $line =~ /^Warning:/) {
	    }
            else {
                carp "Unexpected line: '$line'";
            }
        }

    }

  TOP_FILE:
    foreach my $file (sort {
	                    if ($measure{$b} == $measure{$a}) {
				return $params{$a}->{logC} <=> $params{$b}->{logC};
			    }
			    else {
				return $measure{$b} <=> $measure{$a};
			    }
			 } keys %measure) {
        last TOP_FILE if $top_n == 0;
        print "log(C)=$params{$file}->{logC}, log(gamma)=$params{$file}->{logG}, log(w)=$params{$file}->{logW}: $measure=$measure{$file}\n";
        $top_n--;
	if ($suggest_interval) {
	    my $log_c_min = $params{$file}->{logC} - $stepsize / 2;
	    my $log_c_max = $params{$file}->{logC} + $stepsize / 2;
	    my $log_g_min = $params{$file}->{logG} - $stepsize / 2;
	    my $log_g_max = $params{$file}->{logG} + $stepsize / 2;
	    my $log_w_min = $params{$file}->{logW} - $stepsize / 2;
	    my $log_w_max = $params{$file}->{logW} + $stepsize / 2;
	    print "--log-c-min=$log_c_min --log-c-max=$log_c_max --log-g-min=$log_g_min --log-g-max=$log_g_max --log-w-min=$log_w_min --log-w-max=$log_w_max --stepsize=0.25\n";
	}
    }

    if ($show_progress) {
	my $n_of_files         = scalar @files;
	my $n_of_finished      = scalar keys %measure;
	my $overall_n_of_folds = $n_of_files * $n_of_folds;
	print "Finished $finished_folds/$overall_n_of_folds folds.";
	print " Finished $n_of_finished/$n_of_files parameter combinations.";
	print "\n";
    }
}

sub usage {
    my ($return_code) = @_;

    print << "END";
  Perform distributed grid search for the C, gamma and W hyperparameter of LIBSVM (RBF kernel).
  (c) 2008, 2009, 2010 by Zeno Gantner <zeno.gantner\@gmail.com>

  See "A Practical Guide to Support Vector Classification" by Hsu, Chang and Lin
  for background information.

    usage: $PROGRAM_NAME [OPTIONS] TRAINING_FILE

  options:
    --help                      display this usage information
    --verbose                   increment verbosity level by one
    --email=ADDRESS             send notification about single jobs to email address
    --job-name=NAME             
    --priority=P
    --executable=EXEC           
    --folds=N                   number of folds used for cross validation (default $DEFAULT_N_OF_FOLDS)
    --log-c-min=F
    --log-c-max=F
    --log-g-min=F
    --log-g-max=F
    --log-w-min=F
    --log-w-max=F
    --stepsize=F                set step size of grid search
    --check-results             checks the progress and the results of a started job
                                (assumes the result files to be in the current working directory)
    --measure=MEASURE           rank according to evaluation measure MEASURE (in combination with --check-results)
                                possible values: 'Cross Validation Accuracy' (default), 'nSV'
    --top-n=N                   give out the best N parameter combinations (default $DEFAULT_TOP_N)
    --show-progress
    --suggest-interval
    --result-directory=DIR
    --log-c-values
    --log-g-values
    --log-w-values
    --immediate-start

  WARNING:
    If you want to run several instances of the same script, be sure to start it in different directories or
    use the --result-directory option.
END
    exit $return_code;
}
