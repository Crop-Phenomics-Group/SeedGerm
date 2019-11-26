class SeedPanel:

    def __init__(self, label, centroid, bbox, moments_hu, area, perimeter, eccentricity, major_axis_length,
                 minor_axis_length, solidity, extent, convex_area):
        self.label = label
        self.centroid = centroid
        self.bbox = bbox
        self.moments_hu = moments_hu
        self.area = area
        self.perimeter = perimeter
        self.eccentricity = eccentricity
        self.major_axis_length = major_axis_length
        self.minor_axis_length = minor_axis_length
        self.solidity = solidity
        self.extent = extent
        self.convex_area = convex_area
