package com.broll.picalc;

import java.util.concurrent.TimeUnit;

public class Picalc {
	
	private final static int N=1000000000;

	public static void main(String[] args) {
		long startTime = System.nanoTime();   	
		long pointsInCircle=0;
		for(int i=0; i<N; i++) {
			float x = getRandomPos();
			float y = getRandomPos();
			double distance = Math.sqrt(x*x+y*y);
			if(distance<=1) {
				pointsInCircle++;
			}
		}
		long estimatedTime = System.nanoTime() - startTime;
		double millis = TimeUnit.NANOSECONDS.toMillis(estimatedTime);
		double pi = ((double) pointsInCircle / (double) N) * 4;
		System.out.println("Finished calculating in "+ (millis / 1000d)+" Seconds!");
		System.out.println("=> "+pointsInCircle+" in Circle of");
		System.out.println("=> "+N+" Points");
		System.out.println("=>  PI = "+pi);	
		System.out.println("[PI is = 3.14159265358979323846]");			
	}
	
	private final static float getRandomPos() {
		return (float) (Math.random()*2-1);
	}
	
}
