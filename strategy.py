"""
Reconstructed Pursuer Strategy from main_pyqt.exe
Based on bytecode disassembly of strategy_p.pyc
"""

from shapely.geometry import Point, Polygon, MultiPolygon, LineString
from typing import List
import geometry_lib as gl
from agents import AgentUAV_Evador, AgentUAV_Pursuer
from pulp import *
import numpy as np
import time

LpSolverDefault.msg = 0


class StrategyPursuers:
    def __init__(self, obstacles: List[Polygon], target: Polygon):
        self.inner_buffer_num = -0.01
        self.polygon = [gl.poly_buffer(obs, 0.5) for obs in obstacles]
        self.target = target

        target_vertices = list(target.exterior.coords)[:-1]
        mean_x = sum(v[0] for v in target_vertices) / len(target_vertices)
        mean_y = sum(v[1] for v in target_vertices) / len(target_vertices)
        self.target_center = [mean_x, mean_y]

        if len(self.polygon) > 0:
            self.bigger_multi_polygon = self.polygon[0]
            for i in range(1, len(self.polygon)):
                self.bigger_multi_polygon = self.bigger_multi_polygon.union(self.polygon[i])

            if isinstance(self.bigger_multi_polygon, Polygon):
                self.bigger_multi_polygon = MultiPolygon([self.bigger_multi_polygon])

            self.smaller_polygon = [
                gl.poly_buffer(poly, self.inner_buffer_num)
                for poly in self.bigger_multi_polygon.geoms
            ]

            self.smaller_multi_polygon = self.smaller_polygon[0]
            if len(self.smaller_polygon) > 1:
                for i in range(1, len(self.smaller_polygon)):
                    self.smaller_multi_polygon = self.smaller_multi_polygon.union(self.smaller_polygon[i])

            if isinstance(self.smaller_multi_polygon, Polygon):
                self.smaller_multi_polygon = MultiPolygon([self.smaller_multi_polygon])

            self.vertices_list = [
                Point(vertex)
                for poly in self.bigger_multi_polygon.geoms
                for vertex in list(poly.exterior.coords)[:-1]
            ]

            self.vertexs_num = len(self.vertices_list)

            distanceMatrix = np.zeros([self.vertexs_num + 2, self.vertexs_num + 2])
            for i in range(self.vertexs_num):
                for j in range(i + 1, self.vertexs_num):
                    distanceMatrix[i, j] = gl.point_to_point_without_obstacle(
                        self.vertices_list[i], self.vertices_list[j], self.smaller_multi_polygon
                    )
                    distanceMatrix[j, i] = distanceMatrix[i, j]
            self.distanceMatrix = distanceMatrix

            self.vertexs_to_target = gl.vertexs_to_poly(
                self.vertices_list, self.target, self.smaller_multi_polygon
            )

    def point_to_point(self, start_point, end_point):
        """Find shortest path between two points avoiding obstacles."""
        if len(self.polygon) > 0:
            if self.smaller_multi_polygon.contains(start_point):
                start_point = gl.find_closest_vertex(self.bigger_multi_polygon, start_point)

            if self.smaller_multi_polygon.contains(end_point):
                end_point = gl.find_closest_vertex(self.bigger_multi_polygon, end_point)

            path, dist = gl.getShortestPath(
                start_point, end_point, self.smaller_multi_polygon,
                self.vertices_list, self.distanceMatrix
            )
            return path, dist

        return end_point, end_point.distance(start_point)

    def point_to_target(self, start_point):
        """Find shortest path from point to target."""
        minPoint, minPath, minDist = gl.point_to_poly_with_obs(
            start_point, self.target, self.smaller_multi_polygon,
            self.vertexs_to_target, self.vertices_list, self.distanceMatrix
        )
        return minPoint, minPath, minDist

    def esp_wining(self, agentE: AgentUAV_Evador, agentP: AgentUAV_Pursuer):
        """
        Estimate if pursuer can win against evader.
        Uses distance comparison with velocity ratio.
        """
        positionP = agentP.position
        positionE = agentE.position
        pathPE, distPE = self.point_to_point(positionP, positionE)
        pointET, pathET, distET = self.point_to_target(positionE)
        alphaV = agentP.max_velocity / agentE.max_velocity

        if alphaV <= 1:
            return False

        # Pursuer wins if: distET * alphaV - distPE > 1
        return distET * alphaV - distPE > 1

    def onsite_wining_1V1(self, agentE: AgentUAV_Evador, agentP: AgentUAV_Pursuer):
        """
        Check 1v1 winning using Apollonius circle.
        Returns True if pursuer can capture evader before reaching target.
        """
        positionP = agentP.position
        positionE = agentE.position
        alphaV = agentP.max_velocity / agentE.max_velocity

        if alphaV <= 1:
            return False

        # Get Apollonius circle (capture zone)
        centerA, radiusA = gl.getApolloniusCircle(positionE, positionP, alphaV)
        circleA = Point(centerA).buffer(radiusA)

        # Check if circle intersects obstacles
        if circleA.intersects(self.smaller_multi_polygon):
            return False

        # Check line of sight
        rightSightA, leftSightB = gl.maxCircleSight(positionP, centerA, radiusA)
        if Polygon([positionP, rightSightA, leftSightB]).intersects(self.smaller_multi_polygon):
            return False

        # Pursuer wins if Apollonius circle doesn't intersect target
        return not circleA.intersects(self.target)

    def get_winMatrix(self, agentE_list: List[AgentUAV_Evador], agentP_list: List[AgentUAV_Pursuer]):
        """
        Build win matrix for pursuer-evader matching.
        Higher values = better match for pursuer.
        """
        winMatrix = []
        for j in range(len(agentP_list)):
            winMatrix.append([])

        for i in range(len(agentE_list)):
            agentE = agentE_list[i]
            pointET, pathET, distET = self.point_to_target(Point(agentE.position))
            agentE_list[i].set_path_list(pathET)

            for j in range(len(agentP_list)):
                agentP = agentP_list[j]
                alphaV = agentP.max_velocity / agentE.max_velocity

                if alphaV <= 1:
                    # Pursuer slower - low priority
                    pathPE, distPE = self.point_to_point(Point(agentP.position), Point(agentE.position))
                    winMatrix[j].append(100000.0 - distPE)
                elif self.onsite_wining_1V1(agentE, agentP):
                    # Guaranteed win - highest priority
                    winMatrix[j].append(100000000.0)
                else:
                    pathPE, distPE = self.point_to_point(Point(agentP.position), Point(agentE.position))
                    if distET * (alphaV - 1) - distPE > 1:
                        # Can intercept - high priority
                        winMatrix[j].append(100000000.0 - distPE)
                    else:
                        # Uncertain - medium priority
                        winMatrix[j].append(1000000.0 - distPE)

        return winMatrix

    def esp_match(self, agentE_list: List[AgentUAV_Evador], agentP_list: List[AgentUAV_Pursuer]):
        """
        Match pursuers to evaders using Linear Programming (PULP).
        Maximizes total win probability.
        """
        winMatrix = self.get_winMatrix(agentE_list, agentP_list)

        numAgentE = len(agentE_list)
        numAgentP = len(agentP_list)

        # Create LP problem
        prob = LpProblem("Matching_Problem", LpMaximize)

        # Binary variables for matching
        matches = LpVariable.dicts(
            "Match",
            [(i, j) for i in range(numAgentP) for j in range(numAgentE)],
            cat="Binary"
        )

        # Objective: maximize total win score
        prob += lpSum([
            winMatrix[i][j] * matches[i, j]
            for i in range(numAgentP)
            for j in range(numAgentE)
        ])

        # Constraint: each pursuer matches at most 1 evader
        for i in range(numAgentP):
            prob += lpSum([matches[i, j] for j in range(numAgentE)]) <= 1

        # Constraint: each evader matched by at most 1 pursuer
        for i in range(numAgentE):
            prob += lpSum([matches[j, i] for j in range(numAgentP)]) <= 1

        prob.solve()

        # Extract results
        matchResult = []
        for i in range(numAgentP):
            agentP_list[i].set_target_index(None)
            for j in range(numAgentE):
                if value(matches[i, j]) >= 1:
                    matchResult.append([i, j])
                    agentP_list[i].set_target_index(j)
                    if winMatrix[i][j] > 10000000.0:
                        agentE_list[j].set_would_be_captured(True)
                    else:
                        agentE_list[j].set_would_be_captured(False)

        return matchResult

    def onsilte_capture(self, initCenter, initRadius, E, P, alphaV, captureR, step):
        """
        On-site capture using Apollonius circle pursuit.
        Returns velocity direction for pursuer.
        """
        PE = [E[0] - P[0], E[1] - P[1]]
        normEP = gl.norm_verctor(PE)

        if normEP >= captureR:
            centerA, radiusA = gl.getApolloniusCircle(E, P, alphaV)
            yp = [centerA[0] - initCenter[0], centerA[1] - initCenter[1]]
            rateValue = (initRadius - radiusA + 0.001) / normEP
            zp = [
                rateValue * PE[0] + yp[0] / alphaV,
                rateValue * PE[1] + yp[1] / alphaV
            ]
            normZp = gl.norm_verctor(zp) / alphaV
            Vp = [zp[0] / normZp, zp[1] / normZp]
            return Vp

        return [0, 0]

    def strategy(self, agentE_list: List[AgentUAV_Evador], agentP_list: List[AgentUAV_Pursuer], dt):
        """
        Main strategy: Match pursuers to evaders and pursue.
        Uses two modes:
        1. Path following - pursue evader along shortest path
        2. On-site capture - use Apollonius circle pursuit when close enough
        """
        for agentP in agentP_list:
            if agentP.target_index is None:
                continue

            agentE = agentE_list[agentP.target_index]

            # If target is captured/won, find new target
            if agentE.winning_state or agentE.be_captured:
                # Find living evaders that aren't claimed by others
                livingE_list = [
                    [index, gl.dist_vector(agentE.position, agentP.position)]
                    for index, agentE in enumerate(agentE_list)
                    if not agentE.winning_state and not agentE.be_captured and not agentE.would_be_captured
                ]

                if len(livingE_list) > 0:
                    sorted_livingE_list = sorted(livingE_list, key=lambda x: x[1])
                    agentP.initCenter = None
                    agentP.initRadius = None
                    agentP.set_target_index(sorted_livingE_list[0][0])
                    continue

                # Try any living evader
                livingE_list = [
                    [index, gl.dist_vector(agentE.position, agentP.position)]
                    for index, agentE in enumerate(agentE_list)
                    if not agentE.winning_state and not agentE.be_captured
                ]

                if len(livingE_list) > 0:
                    sorted_livingE_list = sorted(livingE_list, key=lambda x: x[1])
                    agentP.initCenter = None
                    agentP.initRadius = None
                    agentP.set_target_index(sorted_livingE_list[0][0])
                    continue

                # No targets - stop
                agentP.set_velocity([0, 0])
                continue

            # Determine strategy type
            stratey_type = 1  # Default: path following

            # Check if can use on-site capture
            if agentP.initCenter is None:
                if self.onsite_wining_1V1(agentE, agentP):
                    centerA, radiusA = gl.getApolloniusCircle(
                        agentE.position, agentP.position,
                        agentP.max_velocity / agentE.max_velocity
                    )
                    agentP.initCenter = centerA
                    agentP.initRadius = radiusA
                    stratey_type = 2  # On-site capture

            if stratey_type == 1:
                # Path following pursuit
                position_p_Point = Point(agentP.position)
                target_point = agentE_list[agentP.target_index].position
                path, dist = self.point_to_point(position_p_Point, Point(target_point))

                next_Point = path[0]
                velocity_direction = [
                    next_Point.x - position_p_Point.x,
                    next_Point.y - position_p_Point.y
                ]

                if gl.norm_verctor(velocity_direction) < 1e-6:
                    if len(path) > 1:
                        next_Point = path[1]
                        velocity_direction = [
                            next_Point.x - position_p_Point.x,
                            next_Point.y - position_p_Point.y
                        ]

                agentP.set_velocity([
                    velocity_direction[0] / dt,
                    velocity_direction[1] / dt
                ])

            elif stratey_type == 2:
                # On-site Apollonius circle capture
                onsite_velocity = self.onsilte_capture(
                    agentP.initCenter, agentP.initRadius,
                    agentE.position, agentP.position,
                    agentP.max_velocity / agentE.max_velocity,
                    agentP.capture_radius, dt
                )
                agentP.set_velocity([
                    onsite_velocity[0] * agentP.max_velocity,
                    onsite_velocity[1] * agentP.max_velocity
                ])

    def strategy2(self, agentE_list: List[AgentUAV_Evador], agentP_list: List[AgentUAV_Pursuer], dt):
        """
        Alternative greedy matching strategy.
        Matches pursuers to evaders based on distance to target, then distance to pursuer.
        """
        # Filter living evaders
        living_agentE_list = [
            agentE for agentE in agentE_list
            if not agentE.winning_state and not agentE.be_captured
        ]

        if len(living_agentE_list) == 0:
            for agentP in agentP_list:
                agentP.set_velocity([0, 0])
            return

        # Sort evaders by distance to target (closest first - most urgent)
        e_target_list = [
            gl.dist_vector(self.target_center, agentE.position)
            for agentE in living_agentE_list
        ]
        sorted_indices = sorted(range(len(e_target_list)), key=lambda i: e_target_list[i])

        # Greedy matching
        match_result = []
        remain_index_p = list(range(len(agentP_list)))

        for index in sorted_indices:
            position_e = living_agentE_list[index].position
            dist_pe_list = [
                gl.dist_vector(position_e, agentP_list[index_p].position)
                for index_p in remain_index_p
            ]
            sorted_indices_p = sorted(range(len(dist_pe_list)), key=lambda i: dist_pe_list[i])
            match_result.append([remain_index_p[sorted_indices_p[0]], index])
            del remain_index_p[sorted_indices_p[0]]

        # Assign remaining pursuers to closest evaders
        if len(remain_index_p) > 0:
            for indexp in remain_index_p:
                remain_dist_pe_list = [
                    gl.dist_vector(agentE.position, agentP_list[indexp].position)
                    for agentE in living_agentE_list
                ]
                remain_sorted_indices_pe = sorted(
                    range(len(remain_dist_pe_list)),
                    key=lambda i: remain_dist_pe_list[i]
                )
                match_result.append([indexp, remain_sorted_indices_pe[0]])

        # Execute pursuit for each match
        for match in match_result:
            agentP = agentP_list[match[0]]
            agentE = living_agentE_list[match[1]]

            position_p_Point = Point(agentP.position)
            position_e_Point = Point(agentE.position)

            path, dist = self.point_to_point(position_p_Point, position_e_Point)
            next_Point = path[0]
            velocity_direction = [
                next_Point.x - position_p_Point.x,
                next_Point.y - position_p_Point.y
            ]

            if gl.norm_verctor(velocity_direction) < 1e-6:
                if len(path) > 1:
                    next_Point = path[1]
                    velocity_direction = [
                        next_Point.x - position_p_Point.x,
                        next_Point.y - position_p_Point.y
                    ]

            agentP.set_velocity([
                velocity_direction[0] / dt,
                velocity_direction[1] / dt
            ])
