import os
from pathlib import Path

from rich import print

from config import settings
from utils.plots import color_pyplot
from audio_viewer import AudioViewer
from audio_viewer import ScalarOutlineTrack, LabelsZoomedTrack

# Video style from settings.styles
colors = settings.STYLES[settings.OUTPUTS.COLOR_STYLE]
label_colors = colors.LABEL_COLORS
signal_color = colors.WV_SIGNAL
labels_order = settings.OUTPUTS.LABELS_ORDER
density_window_s = settings.OUTPUTS.DENSITY_WINDOW_S
segment_length = settings.OUTPUTS.SEGMENT_LENGTH
fs = settings.AUDIO.SAMPLE_RATE
out_video_width = settings.OUTPUTS.VIDEO_WIDTH


def create_video(
    pred_annotations,
    pred_label_densities,
    t_densities,
    audio,
    input_video,
    output_video,
):

    # Collect densities to plot
    t_list = []
    y_list = []
    styles_list = []
    for label, (y_density_ref, y_density_pred) in pred_label_densities.items():
        t_list.append(t_densities)
        y_list.append(y_density_pred)
        styles_list.append(
            {
                "color": color_pyplot(label_colors[label]),
                "linestyle": "-",
                "label": label,
            }
        )

    # Create AudioViewer
    viewer = AudioViewer(
        audio,
        fs,
        segment_length,
        fig_width=14,
        fig_height=4,
        mpl_style=colors.style_mpl,
    )
    outline_density = ScalarOutlineTrack(t_list, y_list, styles_list)

    zoomed_labels = LabelsZoomedTrack(
        pred_annotations,
        label_colors,
        labels_order=labels_order,
    )

    # Add tracks in top-down order
    viewer.add_track(outline_density)
    viewer.add_track(zoomed_labels)
    frame0 = viewer.update(0)
    # %%
    # Generate animated video from plots above
    temp_video = Path(output_video).with_suffix(".noaudio.mp4")
    viewer.create_video(temp_video)

    # %%
    if input_video.suffix == ".mp4":
        print("[green]Combining classroom and predictions videos...[/green]")
        viewer.combine_videos(
            input_video, temp_video, output_video, out_width=out_video_width
        )
    else:
        print("[green]Creating output video (only classroom audio provided)...[/green]")
        viewer.encode_video(temp_video, input_video, output_video)
    print(f"Removing auxiliary file: {temp_video}")
    os.remove(temp_video)
    print(f"[green]Video ready: {output_video}[/green]")
