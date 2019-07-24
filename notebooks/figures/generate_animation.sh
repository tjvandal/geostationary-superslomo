ffmpeg -r 15 -f image2 -s 1920x1080 -i animation/rbg-%03d.png -crf 25  -pix_fmt yuv420p hurricane.mp4
ffmpeg -r 15 -f image2 -s 1920x1080 -i animation/quiver_plot_band8-ABI-L1b-RadC-%03d.png -crf 25  -pix_fmt yuv420p quiver-animation.mp4

