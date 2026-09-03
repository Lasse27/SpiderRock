import pstats

stats = pstats.Stats("profile_output_2.prof")
stats.sort_stats("cumulative").print_stats(20)  # Top 20 nach kumulativer Zeit