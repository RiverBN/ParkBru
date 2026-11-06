import {test} from 'node:test';
import assert from 'node:assert/strict';
import {getForecast} from '../src/forecast.ts';
test('forecasts change with location and Brunei local time',()=>{
 const noon=Date.parse('2026-10-15T04:00:00Z');
 assert.notDeepEqual(getForecast(noon,0).points,getForecast(noon,1).points);
 assert.notDeepEqual(getForecast(noon,0).points,getForecast(noon+6*3600000,0).points);
 assert.equal(getForecast(noon,0).time,'12:00');
 assert.equal(getForecast(noon,0).points[2].current,true);
});
test('forecast crosses midnight without invalid hours or occupancy',()=>{
 const result=getForecast(Date.parse('2026-10-15T16:30:00Z'),2);
 assert.equal(result.time,'00:30');
 assert.equal(result.points.length,7);
 assert.ok(result.points.every(p=>p.value>=0&&p.value<=100));
 assert.equal(result.points[2].label,'12am');
 assert.match(result.tip,/occupancy|spaces/);
});
