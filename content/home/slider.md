+++
# Slider widget.
widget = "slider"  # See https://sourcethemes.com/academic/docs/page-builder/
headless = true  # This file represents a page section.
active = true  # Activate this widget? true/false
weight = 1  # Order that this section will appear.
height = "300px"

# Slide interval.
# Use `false` to disable animation or enter a time in ms, e.g. `5000` (5s).
interval = 3500


# Slides.
# Duplicate an `[[item]]` block to add more slides.
[[item]]
title = "Interactive visualization"
content = "<br>"
align = "center"  # Choose `center`, `left`, or `right`.

  # Overlay a color or image (optional).
  #   Deactivate an option by commenting out the line, prefixing it with `#`.
  overlay_color = "#666"  # An HTML color value.
  overlay_img = "headers/bubbles-wide.jpg"  # Image path relative to your `static/img/` folder.
  overlay_filter = 0.5  # Darken the image. Value in range 0-1.

  # Call to action button (optional).
  #   Activate the button by specifying a URL and button label below.
  #   Deactivate by commenting out parameters, prefixing lines with `#`.
  cta_label = "Global terrorism"
  cta_url = "#about"
  cta_icon_pack = "fas"
  cta_icon = "chart-line"

[[item]]
  title = "Recent Projects"
  content = "<br>"
  align = "center"

  overlay_color = "#111"  # An HTML color value.
  overlay_img = "headers/mockup-863469_1920.jpg"  # Image path relative to your `static/img/` folder.
  overlay_filter = 0.5  # Darken the image. Value in range 0-1.
  cta_label = "Explore"
  cta_url = "#projects"
  cta_icon_pack = "f1fe"
  cta_icon = "chart-area"

[[item]]
  title = "Analysis highlight"
  content = "<br>"
  align = "center"

  overlay_color = "#333"  # An HTML color value.
  overlay_img = "headers/keyboard-1385706_1280.jpg"  # Image path relative to your `static/img/` folder.
  overlay_filter = 0.5  # Darken the image. Value in range 0-1.
  cta_label = "Explore"
  cta_url = "#gallery"
  cta_icon_pack = "f06e"
  cta_icon = "eye"
+++
