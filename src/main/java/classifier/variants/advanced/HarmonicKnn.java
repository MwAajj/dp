package classifier.variants.advanced;

import classifier.variants.Variant;
import classifier.variants.basic.Knn;
import lombok.Getter;
import lombok.Setter;
import classifier.structure.Structure;
import weka.core.*;

import java.util.*;

@Getter
@Setter
public class HarmonicKnn implements Variant {
    private Structure structure;
    private double[] distances;
    private Instances neighbours;
    Map<Double, Double> info = new HashMap<>();
    private int m;
    private int k;
    private int r;
    private DistanceFunction mDistanceFunction;
    private boolean isLocalMeanSet;


    public HarmonicKnn(Structure structure, int k, int r) {
        this.structure = structure;
        this.k = k;
        this.r = r;
    }

    @Override
    public double[] distributionForInstance(Instance instance, int mNumberClasses) {
        double[] x = hmdKnnAlg(instance, mNumberClasses);
        Utils.normalize(x);
        return x;
    }


    @Override
    public double classifyInstance(Instance target, int mNumberClasses) {
        double min = Double.MAX_VALUE;
        double endClass = 0d;
        double[] meanDistances = hmdKnnAlg(target, mNumberClasses);
        for (int i = 0; i < meanDistances.length; i++) {
            if (meanDistances[i] < min) {
                min = meanDistances[i];
                endClass = i;
            }
        }
        return endClass;
    }

    @Override
    public void setOption(String option) {
        if (Objects.equals(option, "L"))
            isLocalMeanSet = true;
    }

    @Override
    public String getOption() {
        return "-H";
    }

    private double[] hmdKnnAlg(Instance target, int mNumberClasses) {
        if(r > k || r < 0)
            new RuntimeException("Invalid parameter r " + r );
        this.m = mNumberClasses;
        this.neighbours = structure.findKNearestNeighbours(target, k);
        this.distances = structure.getDistances();
        mDistanceFunction.setInstances(this.neighbours);
        sortInstances(neighbours, distances); //1
        Instances meanInstances = getMeanInstances(neighbours); //2
        double[] harmonicMeanDistances = getHarmonicMeanDistances(target, meanInstances);
        if (!isLocalMeanSet) {
            return getResult(harmonicMeanDistances);
        } else {
            return harmonicMeanDistances;
        }
    }

    public void sortInstances(Instances neighbours, double[] distances) {
        Knn.sortNearestInstances(neighbours, distances);
    }

    private double[] getResult(double[] harmonicMeanDistances) {
        double[] sum = new double[m];
        Arrays.fill(sum, 0d);
        for (int i = 0; i < harmonicMeanDistances.length; i++) {
            sum[i] += 1 / harmonicMeanDistances[i];
        }
        for (int i = 0; i < m; i++) {
            sum[i] = k / (sum[i] + 0.00001d);
        }
        return sum;
    }

    private double[] getHarmonicMeanDistances(Instance target, Instances meanVectors) {
        double[] harmonicMeanDistances = new double[m];
        double[] denominator = new double[m];
        for (int i = 0; i < m; i++) {
            double distance = mDistanceFunction.distance(target, meanVectors.get(i));
            denominator[i] += distance;
        }
        for (int i = 0; i < m; i++) {
            harmonicMeanDistances[i] = r / denominator[i];
        }
        return harmonicMeanDistances;
    }

    private Instances getMeanInstances(Instances neighbours) {
        Instances inst = new Instances("target", getALlAttributes(neighbours.firstInstance()), m);
        double[][] meanVectors = new double[m][neighbours.numAttributes()];
        int[] counts = new int[m];
        Arrays.fill(counts, 0);
        for (int i = 0; i < neighbours.size(); i++) {
            for (int j = 0; j < neighbours.numAttributes(); j++) {
                meanVectors[(int) neighbours.get(i).classValue()][j] += neighbours.get(i).value(j);
            }
            counts[(int) neighbours.get(i).classValue()] += 1;
        }
        for (int i = 0; i < meanVectors.length; i++) {
            for (int j = 0; j < neighbours.numAttributes(); j++) {
                if (counts[i] == 0)
                    meanVectors[i][j] = (1 / (counts[i] + 0.0001d)) * meanVectors[i][j]; //avoid to div zero
                else meanVectors[i][j] = (1d / counts[i]) * meanVectors[i][j];
            }
            inst.add(new DenseInstance(1d, meanVectors[i]));
        }
        return inst;
    }


    private ArrayList<Attribute> getALlAttributes(Instance instance) {
        ArrayList<Attribute> attr = new ArrayList<>();
        for (int i = 0; i < instance.numAttributes(); i++) {
            attr.add(instance.attribute(i));
        }
        return attr;
    }
}
