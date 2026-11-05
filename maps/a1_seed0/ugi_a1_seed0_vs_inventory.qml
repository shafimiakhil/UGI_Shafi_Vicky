<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28.0" styleCategories="AllStyleCategories">
  <pipe>
    <rasterrenderer type="paletted" band="1" opacity="0.6" alphaBand="-1" nodataColor="">
      <rasterTransparency/>
      <colorPalette>
        <paletteEntry value="1" alpha="255" color="#2b6cd4" label="In the inventory and mapped"/>
        <paletteEntry value="2" alpha="255" color="#e8743b" label="Mapped outside the inventory"/>
        <paletteEntry value="3" alpha="255" color="#19a979" label="In the inventory, not mapped"/>
        <paletteEntry value="4" alpha="255" color="#969696" label="Mapped outside the ejido (no reference)"/>
      </colorPalette>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0" gamma="1"/>
    <huesaturation colorizeOn="0" colorizeRed="255" colorizeGreen="128" colorizeBlue="128" colorizeStrength="100" grayscaleMode="0" saturation="0" invertColors="0"/>
    <rasterresampler maxOversampling="2"/>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
