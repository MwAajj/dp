package classifier.variants.advanced;

import classifier.variants.Variant;
import lombok.Getter;
import lombok.Setter;
import structure.Structure;
import weka.core.Instance;
import weka.core.Instances;

import java.util.HashMap;
import java.util.Map;

@Getter
@Setter
public class HarmonicKnn implements Variant {
    private Structure structure;
    private double[] distances;
    Map<Double, Double> info = new HashMap<>();
    private int m;
    private int k;
    private int r;

    public HarmonicKnn(Structure structure, int k, int r) {
        if (r > k)
            throw new RuntimeException("r parameter must be smaller then k");
        this.structure = structure;
        this.k = k;
        this.r = r;
    }

    @Override
    public double[] distributionForInstance(Instance instance, int m_NumClasses) {
        this.m = m_NumClasses;
        return new double[1];
    }

    @Override
    public double classifyInstance(Instance target, int m_NumClasses) {
        this.m = m_NumClasses;
        double min = Double.MAX_VALUE, endClass = 0d;
        Instances neighbours = structure.findKNearestNeighbours(target, k);
        this.distances = structure.getDistances(); //1
        double[] meanVectors = getMeanVectors(neighbours); //2
        double[] harmonicMeanDistances = getHarmonicMeanDistances(neighbours, target, meanVectors); //3


        printInfo();
        return endClass;
    }

    private double[] getHarmonicMeanDistances(Instances neighbours, Instance target, double[] meanVectors) {
        double[] harmonicMeanDistances = new double[neighbours.size()];
        double[] denominator = new double[neighbours.size()];
        for (int i = 0; i < neighbours.size(); i++) {
            denominator[i] = 1 / (meanVectors[i]);
        }
        return harmonicMeanDistances;
    }

    @Override
    public String getOption() {
        return null;
    }


    private double[] getMeanVectors(Instances neighbours) {
        double[] meanVectors = new double[m];
        for (int i = 0; i < neighbours.size(); i++) {
            meanVectors[(int) neighbours.get(i).classValue()] += distances[i];
        }
        for (int i = 0; i < meanVectors.length; i++) {
            meanVectors[i] = (1 / (double) k) * meanVectors[i];
        }
        return meanVectors;
    }

    private void printInfo() {
        System.out.println("\n--------------------HMD-KNN------------");
        System.out.println(info);
        System.out.println("-------------------------------------");
    }
}
